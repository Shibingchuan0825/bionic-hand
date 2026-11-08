// HandSceneBuilder.cs
//
// One-click scene construction for the bionic hand.
//
// Why per-part files
// ------------------
// Unity's OBJ importer merges the ``o`` object groups of a single file into one
// mesh, so a 34-object hand.obj arrives as a single blob with no separable
// parts -- there is nothing to parent to a joint.  The exporter therefore also
// writes one OBJ per rigid part, and this builder loads those: each file
// becomes its own GameObject, which is exactly the granularity the rig needs.
//
// Why the joints are inferred
// ---------------------------
// The source model has no URDF, so joint axes and origins were reconstructed
// from the geometry (see docs/模型分析_2026-10-18.md).  They are a starting
// point, not measured fact: the first run must be checked joint by joint.  A
// finger that bends the wrong way needs Invert; one that swings sideways means
// that joint's inferred axis is wrong.
//
// What it does NOT do
// -------------------
// It does not know which way a joint should bend, and it does not set limits to
// anything physical.  Every joint gets the same demonstration limit.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace BionicHand.EditorTools
{
    [Serializable]
    public class PartJson
    {
        public int index;
        public string name;
        public string part_file;
        public float[] centre_mm;
    }

    [Serializable]
    public class HandDocument
    {
        public PartJson[] parts;
    }

    [Serializable]
    public class JointJson
    {
        public string finger;
        public string joint;
        public string name;
        public float[] axis;
        public float[] origin_mm;
        public string driven_part_name;
        public int driven_part_index;
        public string obj_object_name;
        public string confidence;
        public float extended_angle;
        public float flexed_angle;
    }

    [Serializable]
    public class JointDocument
    {
        public JointJson[] joints;
        public float obj_scale;
    }

    /// One axis of a camera calibration capture.
    ///
    /// An array of these rather than a dictionary because Unity's JsonUtility
    /// cannot deserialize a JSON object into a Dictionary.
    [Serializable]
    public class SensorAxisJson
    {
        public string name;
        public float min;
        public float max;
        public float span;
    }

    [Serializable]
    public class SensorCalibrationJson
    {
        public string source;
        public SensorAxisJson[] axes;
    }

    public static class HandSceneBuilder
    {
        private const string PartsDir = "Assets/Models/Parts";
        private const string HandJson = "Assets/Models/hand.json";
        private const string JointsJson = "Assets/Models/joints_inferred.json";
        private const string RootName = "BionicHand";

        // The range the Python mapper actually produces.  Kept in sync with
        // EXTENDED_ANGLE / FLEXED_ANGLE in src/ui/virtual_hand.py.
        // Measured from the operator's own hand with tools/measure_tracking.py:
        // a fist reads about 124 degrees and an open hand about 179.  The
        // protocol's 0..180 span is far wider than a real hand ever produces,
        // so calibrating to it wastes most of the model's travel -- a fist
        // moved the joints only 38% of the way, which is why the bend looked
        // feeble.
        private const float ExtendedAngle = 179f;
        private const float FlexedAngle = 124f;

        /// Written by tools/measure_tracking.py --write-calibration.  Optional:
        /// without it the modelled joint limits above are used, which is what
        /// the scene had before anyone measured the camera.
        private const string CalibrationJson = "Assets/Models/sensor_calibration.json";

        /// Below this observed span the axis was not exercised by the capture,
        /// so its extremes say more about the operator's range of motion in
        /// those 15 seconds than about the joint.
        private const float MinCalibrationSpan = 10f;

        [MenuItem("Bionic Hand/Build Hand Scene", priority = 0)]
        public static void Build()
        {
            string projectRoot = Path.Combine(Application.dataPath, "..");
            string partsAbsolute = Path.Combine(projectRoot, PartsDir);

            if (!Directory.Exists(partsAbsolute))
            {
                Debug.LogError(
                    $"[HandSceneBuilder] {PartsDir} not found. " +
                    "Copy the whole unity/Assets/ tree into the project, " +
                    "including Models/Parts/.");
                return;
            }
            if (!File.Exists(Path.Combine(projectRoot, HandJson)) ||
                !File.Exists(Path.Combine(projectRoot, JointsJson)))
            {
                Debug.LogError(
                    $"[HandSceneBuilder] missing hand.json or joints_inferred.json " +
                    $"under Assets/Models/.");
                return;
            }

            var hand = JsonUtility.FromJson<HandDocument>(
                File.ReadAllText(Path.Combine(projectRoot, HandJson)));
            var joints = JsonUtility.FromJson<JointDocument>(
                File.ReadAllText(Path.Combine(projectRoot, JointsJson)));

            if (hand?.parts == null || hand.parts.Length == 0)
            {
                Debug.LogError("[HandSceneBuilder] hand.json lists no parts.");
                return;
            }
            if (joints?.joints == null || joints.joints.Length == 0)
            {
                Debug.LogError("[HandSceneBuilder] joints_inferred.json lists no joints.");
                return;
            }

            var sensorCalibration = LoadSensorCalibration(projectRoot);

            var existing = GameObject.Find(RootName);
            if (existing != null)
            {
                if (!Application.isBatchMode)
                {
                    if (!EditorUtility.DisplayDialog(
                            "Bionic Hand",
                            $"A '{RootName}' object already exists. Replace it?",
                            "Replace", "Cancel"))
                    {
                        return;
                    }
                }
                UnityEngine.Object.DestroyImmediate(existing);
            }

            var root = new GameObject(RootName);

            // Everything visual hangs off the wrist, so a single transform
            // carries pronation/supination for the whole hand.  The model had
            // no wrist joint at all, so the wrist_rot axis the Python side
            // sends was reaching nothing.
            // The wrist pivot sits at the real wrist, not at the model origin.
            // Parts below the split are the forearm/base assembly and must NOT
            // turn with the hand -- parenting everything to the wrist made the
            // rotation read as "the base is spinning" rather than as pronation.
            const float WristSplitMm = 75f;

            var baseRoot = new GameObject("Base");
            baseRoot.transform.SetParent(root.transform, false);

            var wrist = new GameObject("Wrist");
            wrist.transform.SetParent(root.transform, false);
            // Pivot at the wrist itself (z = split), not at the model origin.
            wrist.transform.localPosition = new Vector3(0f, 0f, WristSplitMm * 0.001f);

            var model = new GameObject("Hand");
            model.transform.SetParent(wrist.transform, false);

            // --- instantiate every part ---------------------------------------
            var partsByName = new Dictionary<string, GameObject>();
            int missing = 0;
            foreach (var part in hand.parts)
            {
                string assetPath = $"{PartsDir}/{part.part_file}";
                var asset = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
                if (asset == null)
                {
                    missing++;
                    Debug.LogWarning($"[HandSceneBuilder] could not load {assetPath}");
                    continue;
                }
                // Plain copy rather than a prefab instance: the parts are then
                // ordinary scene objects the operator can select, inspect and
                // move, instead of blue prefab links that resist editing.
                var go = (GameObject)UnityEngine.Object.Instantiate(asset);
                bool isBase = part.centre_mm != null &&
                              part.centre_mm.Length >= 3 &&
                              part.centre_mm[2] < WristSplitMm;
                // Hand parts keep their world position while becoming children
                // of the offset wrist, so the mesh stays put and only the pivot
                // moves.  Base parts stay at the root.
                go.transform.SetParent(
                    isBase ? baseRoot.transform : model.transform,
                    worldPositionStays: !isBase);
                go.name = Path.GetFileNameWithoutExtension(part.part_file);
                partsByName[go.name] = go;
            }
            Debug.Log($"[HandSceneBuilder] parts loaded: {partsByName.Count}, missing: {missing}");

            // --- build the joint chains ---------------------------------------
            float scale = joints.obj_scale > 0f ? joints.obj_scale : 0.001f;

            // The thumb's chain is cmc -> mcp -> ip, the fingers' is
            // mcp -> pip -> dip.  A single shared order array put the thumb's
            // mcp above its cmc, producing the impossible chain
            // mcp -> cmc -> ip.
            var fingerOrder = new[] { "mcp", "pip", "dip" };
            var thumbOrder = new[] { "cmc", "mcp", "ip" };

            var byFinger = joints.joints
                .GroupBy(j => j.finger)
                .ToDictionary(g => g.Key, g => g.ToList());

            var jointTransforms = new Dictionary<string, Transform>();
            int created = 0, unbound = 0;

            foreach (var pair in byFinger)
            {
                var order = pair.Key == "thumb" ? thumbOrder : fingerOrder;
                var chain = pair.Value
                    .OrderBy(j =>
                    {
                        int at = Array.IndexOf(order, j.joint);
                        return at >= 0 ? at : 99;
                    })
                    .ToList();

                Transform parent = wrist.transform;
                // The chain's positions are absolute, so the first joint must
                // be expressed relative to the offset wrist.
                Vector3 parentOrigin = wrist.transform.localPosition;

                foreach (var spec in chain)
                {
                    Vector3 origin = ToVector(spec.origin_mm) * scale;

                    var jointGo = new GameObject(spec.name);
                    jointGo.transform.SetParent(parent, false);
                    jointGo.transform.localPosition = origin - parentOrigin;
                    jointGo.transform.localRotation = Quaternion.identity;
                    jointTransforms[spec.name] = jointGo.transform;

                    // Move the driven part under its joint WITHOUT moving it.
                    // The mesh vertices are already at assembly coordinates, so
                    // worldPositionStays re-bases localPosition and the part now
                    // rotates about the joint.
                    if (!string.IsNullOrEmpty(spec.obj_object_name) &&
                        partsByName.TryGetValue(spec.obj_object_name, out var part))
                    {
                        part.transform.SetParent(jointGo.transform, true);
                    }
                    else
                    {
                        unbound++;
                        Debug.LogWarning(
                            $"[HandSceneBuilder] {spec.name}: no part named " +
                            $"'{spec.obj_object_name}' in the model.");
                    }

                    parent = jointGo.transform;
                    parentOrigin = origin;
                    created++;
                }
            }

            // --- components ----------------------------------------------------
            var receiver = root.AddComponent<PoseReceiver>();
            var rig = root.AddComponent<HandRig>();
            root.AddComponent<StatusHud>();

            rig.receiver = receiver;
            rig.autoBindByChildName = true;
            // Each joint carries its own measured range: the MCP joints move
            // only 20-35 degrees while the PIPs move 65-154, so one shared pair
            // either leaves the knuckles almost still or makes the distal
            // joints wildly overshoot.
            rig.useGlobalCalibration = false;
            rig.joints = new List<JointBinding>();

            foreach (var spec in joints.joints)
            {
                if (!jointTransforms.TryGetValue(spec.name, out var jointTransform)) continue;

                float extendedAngle = spec.extended_angle > 0f
                    ? spec.extended_angle : ExtendedAngle;
                float flexedAngle = spec.flexed_angle > 0f
                    ? spec.flexed_angle : FlexedAngle;

                // The modelled limits describe the mechanism, not what the
                // camera can see.  thumb_cmc is modelled 179 -> 132 but only
                // ever measures 132 -> 158, so driving it on the modelled
                // range pinned the thumb near half-flexed across its whole
                // travel and it looked like it never moved.
                if (sensorCalibration.TryGetValue(spec.name, out var calibration))
                {
                    extendedAngle = calibration.max;
                    flexedAngle = calibration.min;
                }

                rig.joints.Add(new JointBinding
                {
                    axisName = spec.name,
                    joint = jointTransform,
                    rotationAxis = ToVector(spec.axis).normalized,
                    extendedAngle = extendedAngle,
                    flexedAngle = flexedAngle,
                    maxFlexionDegrees = spec.finger == "thumb" ? 70f : 90f,
                    invert = false,
                    // The 12-channel protocol has no DIP axis, so the distal
                    // joint follows the PIP -- the same coupling the Python
                    // renderer uses, and what a single-tendon finger does.
                    coupleTo = CouplingSource(spec),
                    coupleFactor = CouplingFactor(spec),
                });
            }

            // The wrist.  90 degrees is neutral and the value is a signed offset:
            // this is pronation/supination, which turns about the forearm axis
            // (+Z in this model), not flexion.
            rig.joints.Add(new JointBinding
            {
                axisName = "wrist_rot",
                joint = wrist.transform,
                rotationAxis = new Vector3(0f, 0f, 1f),
                centeredOnNeutral = true,
                maxFlexionDegrees = 90f,
            });

            EditorSceneManager.MarkAllScenesDirty();
            Selection.activeGameObject = root;

            EnsureCameraAndLight(root);

            // In batch mode nothing saves before -quit, so the build would be
            // discarded.  In the editor the operator decides when to save.
            if (Application.isBatchMode)
            {
                const string scenePath = "Assets/Scenes/HandScene.unity";
                Directory.CreateDirectory(
                    Path.GetDirectoryName(Path.Combine(projectRoot, scenePath)));
                bool saved = EditorSceneManager.SaveScene(
                    UnityEngine.SceneManagement.SceneManager.GetActiveScene(), scenePath);
                Debug.Log($"[HandSceneBuilder] scene saved to {scenePath}: {saved}");
            }

            Debug.Log(
                $"[HandSceneBuilder] DONE joints={created} bindings={rig.joints.Count} " +
                $"unbound={unbound}. " +
                "NEXT: press Play and run 'python tools/unity_pose_test.py send'. " +
                "Check every joint -- wrong direction needs Invert, sideways motion " +
                "means the inferred axis is wrong for that joint.");
        }

        /// <summary>
        /// A joint that should follow another axis instead of its own, or "".
        ///
        /// The DIP has no channel of its own, so it follows the PIP.  A narrow
        /// MCP is coupled too: its measured span is under 40 degrees, which
        /// needs a 2.8x visual gain and therefore amplifies tracking noise just
        /// as much.  In a fist the knuckle and the PIP flex together anyway, so
        /// following the PIP is both steadier and anatomically reasonable.
        /// </summary>
        /// Load the camera calibration, keyed by axis name.  Missing file or
        /// under-exercised axes are simply absent, so callers fall back.
        private static Dictionary<string, SensorAxisJson> LoadSensorCalibration(
            string projectRoot)
        {
            var map = new Dictionary<string, SensorAxisJson>();
            var path = Path.Combine(projectRoot, CalibrationJson);
            if (!File.Exists(path))
            {
                Debug.Log(
                    $"[HandSceneBuilder] no sensor calibration at {CalibrationJson}; " +
                    "joints will be driven from the modelled limits.  Run " +
                    "tools/measure_tracking.py --write-calibration to produce one.");
                return map;
            }

            var doc = JsonUtility.FromJson<SensorCalibrationJson>(File.ReadAllText(path));
            if (doc?.axes == null) return map;

            foreach (var axis in doc.axes)
            {
                if (axis == null || string.IsNullOrEmpty(axis.name)) continue;
                if (axis.span < MinCalibrationSpan) continue;
                map[axis.name] = axis;
            }
            Debug.Log($"[HandSceneBuilder] sensor calibration: {map.Count} axes " +
                      $"from {doc.source}");
            return map;
        }

        private static string CouplingSource(JointJson spec)
        {
            if (spec.joint == "dip") return spec.name.Replace("_dip", "_pip");

            // Only the four long fingers.  The thumb's chain is CMC/MCP/IP, so
            // "replace _mcp with _pip" produces thumb_pip -- an axis that does
            // not exist, which silently freezes the thumb.
            if (spec.finger == "thumb") return "";

            float span = spec.extended_angle - spec.flexed_angle;
            if (spec.joint == "mcp" && spec.extended_angle > 0f && span < 40f)
            {
                return spec.name.Replace("_mcp", "_pip");
            }
            return "";
        }

        private static float CouplingFactor(JointJson spec)
        {
            if (spec.joint == "dip") return 0.75f;
            if (spec.finger == "thumb") return 1f;

            float span = spec.extended_angle - spec.flexed_angle;
            if (spec.joint == "mcp" && spec.extended_angle > 0f && span < 40f)
            {
                return 0.9f;
            }
            return 1f;
        }

        private static Vector3 ToVector(float[] values)
        {
            if (values == null || values.Length < 3) return Vector3.zero;
            return new Vector3(values[0], values[1], values[2]);
        }

        [MenuItem("Bionic Hand/Verify Scene", priority = 2)]
        public static void Verify()
        {
            const string scenePath = "Assets/Scenes/HandScene.unity";
            if (!File.Exists(Path.Combine(Application.dataPath, "../" + scenePath)))
            {
                Debug.LogError($"[Verify] {scenePath} not found");
                return;
            }

            var scene = EditorSceneManager.OpenScene(scenePath);
            var root = GameObject.Find(RootName);
            if (root == null)
            {
                Debug.LogError($"[Verify] no '{RootName}' in {scenePath}");
                return;
            }

            var receiver = root.GetComponent<PoseReceiver>();
            var rig = root.GetComponent<HandRig>();
            var hud = root.GetComponent<StatusHud>();

            Debug.Log($"[Verify] PoseReceiver present: {receiver != null}");
            Debug.Log($"[Verify] HandRig present: {rig != null}");
            Debug.Log($"[Verify] StatusHud present: {hud != null}");

            if (rig == null)
            {
                Debug.LogError(
                    "[Verify] HandRig is MISSING -- the scene references a script " +
                    "GUID that does not match Assets/Scripts/HandRig.cs.meta. " +
                    "The model is fine; the components need re-attaching.");
                return;
            }

            int bound = 0, nullJoints = 0;
            foreach (var b in rig.joints)
            {
                if (b.joint != null) bound++;
                else nullJoints++;
            }
            Debug.Log($"[Verify] bindings={rig.joints.Count} bound={bound} null={nullJoints}");

            int meshCount = 0;
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
            {
                if (t.GetComponent<MeshFilter>() != null) meshCount++;
            }
            Debug.Log($"[Verify] renderable meshes under root: {meshCount}");
            Debug.Log($"[Verify] scene '{scene.name}' roots={scene.rootCount}");
        }

        /// <summary>
        /// Create a camera and a light framed on the hand.
        ///
        /// A scene built from scratch has neither, which is why the Game view
        /// reported "No camera rendering" and why an unlit model is invisible.
        /// The camera looks down -Y: the palm's normal is Y, so flexion happens
        /// in the XZ plane and is seen face-on, which is what makes a wrong
        /// joint axis obvious.
        /// </summary>
        private static void EnsureCameraAndLight(GameObject root)
        {
            Vector3 centre = Vector3.zero;
            float size = 0.3f;

            var renderers = root.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length > 0)
            {
                Bounds b = renderers[0].bounds;
                foreach (var r in renderers) b.Encapsulate(r.bounds);
                centre = b.center;
                size = Mathf.Max(b.size.x, Mathf.Max(b.size.y, b.size.z));
            }

            // Distance so the hand fills roughly two thirds of a 60-degree view.
            float distance = (size / 0.65f) / (2f * Mathf.Tan(Mathf.Deg2Rad * 30f));

            var cameraGo = GameObject.Find("Main Camera");
            if (cameraGo == null)
            {
                cameraGo = new GameObject("Main Camera");
                cameraGo.tag = "MainCamera";
                cameraGo.AddComponent<Camera>();
                cameraGo.AddComponent<AudioListener>();
            }
            var camera = cameraGo.GetComponent<Camera>();
            camera.fieldOfView = 60f;
            // Depth-buffer precision is governed by the near/far RATIO, not the
            // absolute values.  An earlier setting of 0.01..1000 gave a ratio of
            // 1:100000, which is so coarse that the hand's overlapping shells
            // (palm panels, joint rings, caps) z-fight and shimmer.  0.05..5 is
            // a 1:100 ratio and still comfortably contains a 0.23 m model
            // viewed from ~0.3 m.
            camera.nearClipPlane = 0.05f;
            camera.farClipPlane = 5f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.15f, 0.16f, 0.18f);

            cameraGo.transform.position = centre + new Vector3(0f, distance, 0f);
            // +Y down, with +Z (the finger direction) as screen-up.
            cameraGo.transform.rotation = Quaternion.Euler(90f, 0f, 0f);

            var lightGo = GameObject.Find("Directional Light");
            if (lightGo == null)
            {
                lightGo = new GameObject("Directional Light");
                var light = lightGo.AddComponent<Light>();
                light.type = LightType.Directional;
                light.intensity = 1.1f;
            }
            lightGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            Debug.Log($"[HandSceneBuilder] camera at {cameraGo.transform.position}, " +
                      $"looking down -Y, distance {distance:F3} m for a {size:F3} m model");
        }

        [MenuItem("Bionic Hand/Capture Preview", priority = 3)]
        public static void CapturePreview()
        {
            const string scenePath = "Assets/Scenes/HandScene.unity";
            EditorSceneManager.OpenScene(scenePath);

            var camera = Camera.main;
            if (camera == null)
            {
                Debug.LogError("[Capture] no Main Camera in the scene");
                return;
            }

            const int width = 800, height = 600;
            var rt = new RenderTexture(width, height, 24);
            var previous = camera.targetTexture;
            camera.targetTexture = rt;
            camera.Render();

            RenderTexture.active = rt;
            var texture = new Texture2D(width, height, TextureFormat.RGB24, false);
            texture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            texture.Apply();
            RenderTexture.active = null;
            camera.targetTexture = previous;

            string path = Path.Combine(Application.dataPath, "../preview.png");
            File.WriteAllBytes(path, texture.EncodeToPNG());
            Debug.Log($"[Capture] wrote {Path.GetFullPath(path)}");
        }

        [MenuItem("Bionic Hand/Report Rig Setup", priority = 1)]
        public static void Report()
        {
            var root = GameObject.Find(RootName);
            if (root == null)
            {
                Debug.LogError("[HandSceneBuilder] no BionicHand object in the scene.");
                return;
            }
            var rig = root.GetComponent<HandRig>();
            var receiver = root.GetComponent<PoseReceiver>();
            Debug.Log($"[HandSceneBuilder] bindings: {rig?.joints?.Count ?? 0}, " +
                      $"receiver port: {receiver?.port}");
        }
    }
}
