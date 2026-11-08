// HandRig.cs
//
// Drives a hand model's joint transforms from the pose stream.
//
// Angle convention
// ----------------
// The bridge sends joint angles in degrees where 180 = fully extended and 0 =
// fully flexed.  The tracker, however, never emits those extremes: measuring
// the interior angle at a joint, a straight finger reads about 158 and a closed
// one about 68.  Each binding therefore carries its own extendedAngle /
// flexedAngle pair (defaults match the Python side) and the value is remapped
// between them, so the model uses its full range of motion over the range the
// hand actually produces.
//
// Getting this wrong is subtle: calibrating against 0-180 instead of 68-158
// makes a closing hand move the model only a third as far, which reads as
// "the model is ignoring my hand" rather than as an obvious bug.
//
// Rigging
// -------
// This expects a RIGID hierarchy, not a skinned mesh: each phalanx is its own
// object, parented to an empty whose local origin sits on the joint's axis of
// rotation.  Rotating that empty swings the phalanx.  A mechanical hand is made
// of rigid bodies, so no skinning or weight painting is needed -- and rigid
// parenting is more accurate than skinning would be.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace BionicHand
{
    /// One joint of the model, bound to one axis of the pose stream.
    [Serializable]
    public class JointBinding
    {
        [Tooltip("Axis name from the pose stream, e.g. index_mcp")]
        public string axisName;

        [Tooltip("Transform whose LOCAL rotation is driven. Its origin must sit " +
                 "on the joint's axis of rotation.")]
        public Transform joint;

        [Tooltip("Local axis to rotate about. For a finger knuckle this is " +
                 "usually the axis running across the hand.")]
        public Vector3 rotationAxis = Vector3.right;

        [Tooltip("Stream angle that means fully extended.")]
        public float extendedAngle = 158f;

        [Tooltip("Stream angle that means fully flexed.")]
        public float flexedAngle = 68f;

        [Tooltip("How far this joint rotates at full flexion, in degrees.")]
        public float maxFlexionDegrees = 90f;

        [Tooltip("Flip the direction of rotation.")]
        public bool invert;

        [Tooltip("Drive this joint from a DIFFERENT axis. The 12-channel protocol " +
                 "carries no DIP, so the distal joints follow the PIP instead -- " +
                 "which is also what a single-tendon finger does mechanically.")]
        public string coupleTo = "";

        [Tooltip("Fraction of the source joint's flexion to apply when coupled.")]
        [Range(0f, 2f)] public float coupleFactor = 0.75f;

        [Tooltip("Treat the angle as a signed offset from 90 instead of 0..180 " +
                 "flexion. Used by the wrist, where 90 is neutral and the axis " +
                 "carries pronation/supination.")]
        public bool centeredOnNeutral;

        [HideInInspector] public Quaternion restRotation;
        [HideInInspector] public float appliedAngle = float.NaN;
        // The rotation for appliedAngle.  Kept so the joint can keep converging
        // on it while new readings fall inside the deadband.
        [HideInInspector] public Quaternion targetRotation = Quaternion.identity;
    }

    public enum WristSource
    {
        Axis,
        Orientation,
    }

    public class HandRig : MonoBehaviour
    {
        [Tooltip("Where pose data comes from. Found automatically if left empty.")]
        public PoseReceiver receiver;

        public List<JointBinding> joints = new List<JointBinding>();

        [Header("Response")]
        [Tooltip("Seconds to blend toward a new pose. 0 = snap.")]
        [Range(0f, 0.5f)] public float smoothing = 0.05f;

        [Header("Calibration")]
        [Tooltip("Use the two values below for every joint instead of each " +
                 "binding's own pair. One place to tune after measuring your hand.")]
        public bool useGlobalCalibration = true;

        [Tooltip("Stream angle that means fully extended. Measured 179 for a " +
                 "typical open hand.")]
        public float globalExtendedAngle = 179f;

        [Tooltip("Stream angle that means fully flexed. Measured 124 for a " +
                 "typical fist.")]
        public float globalFlexedAngle = 124f;

        [Tooltip("Multiplier on the flexion angle. Raise this if a fist only " +
                 "bends the model a little: the mapper's real span is usually " +
                 "narrower than the 0..180 protocol range, so the model uses " +
                 "only part of its travel.")]
        [Range(0.5f, 4f)] public float flexionGain = 1.4f;

        [Tooltip("Ignore angle changes smaller than this, in degrees. Stops the " +
                 "joints twitching on floating-point noise while a pose is held.")]
        [Range(0f, 5f)] public float angleDeadband = 1.5f;

        [Tooltip("After this long with no packet, treat the pose as stale.")]
        public float staleAfterSeconds = 1.0f;

        [Tooltip("Return every joint to its rest pose when the stream goes stale.")]
        public bool relaxWhenStale = true;

        [Header("Wrist")]
        [Tooltip("Axis: drive the wrist from the wrist_rot stream angle. " +
                 "Orientation: flip the hand front/back from the recognised " +
                 "orientation. The orientation route is far more stable -- the " +
                 "measured wrist_rot is noisy (std ~28 degrees).")]
        public WristSource wristSource = WristSource.Orientation;

        [Tooltip("Wrist angle when the palm faces the camera.")]
        public float frontWristDegrees = 0f;

        [Tooltip("Wrist angle when the edge of the hand faces the camera.")]
        public float sideWristDegrees = 90f;

        [Tooltip("Wrist angle when the back of the hand faces the camera. " +
                 "Set to 0 and 180 swapped if the flip looks reversed.")]
        public float backWristDegrees = 180f;

        [Header("Orientation feedback")]
        [Tooltip("Tint the whole hand by the recognised orientation. Front, back " +
                 "and side then look different at a glance, and a wrong " +
                 "front/back call is obvious instead of subtle.")]
        public bool tintByOrientation = false;
        public Color frontColor = new Color(0.45f, 0.95f, 0.55f);
        public Color backColor = new Color(0.40f, 0.62f, 1.00f);
        public Color sideColor = new Color(0.85f, 0.85f, 0.85f);
        public Color unknownColor = new Color(0.95f, 0.55f, 0.20f);

        [Header("Setup helpers")]
        [Tooltip("Match joints to child transforms by name, e.g. a child called " +
                 "'index_mcp' binds automatically.")]
        public bool autoBindByChildName = true;

        private readonly Dictionary<string, JointBinding> _byName =
            new Dictionary<string, JointBinding>();
        private Renderer[] _renderers;
        private MaterialPropertyBlock _tintBlock;
        private string _lastOrientation;

        private string _stableOrientation;
        private string _pendingOrientation;
        private int _pendingOrientationFrames;

        [Header("Orientation stability")]
        [Tooltip("Frames a new orientation must persist before the wrist follows it. " +
                 "The classifier is hysteretic, but the stream carries a discretised " +
                 "three-state answer, and each flip near the edge-on boundary swings " +
                 "the wrist through 90 degrees.")]
        public int wristHoldFrames = 15;

        void Awake()
        {
            if (receiver == null) receiver = FindObjectOfType<PoseReceiver>();

            if (autoBindByChildName) AutoBind();
            CacheRestRotations();
            BuildLookup();

            _renderers = GetComponentsInChildren<Renderer>(true);
            _tintBlock = new MaterialPropertyBlock();
        }

        /// Bind any joint with no transform assigned to a child of the same name.
        /// Saves assigning a dozen transforms by hand, and keeps the binding
        /// honest: the name in the stream and the name in the hierarchy match.
        public void AutoBind()
        {
            var byName = new Dictionary<string, Transform>();
            foreach (var t in GetComponentsInChildren<Transform>(true))
            {
                if (!byName.ContainsKey(t.name)) byName[t.name] = t;
            }

            foreach (var binding in joints)
            {
                if (binding.joint != null || string.IsNullOrEmpty(binding.axisName)) continue;
                if (byName.TryGetValue(binding.axisName, out var match))
                {
                    binding.joint = match;
                }
            }
        }

        private void CacheRestRotations()
        {
            foreach (var binding in joints)
            {
                if (binding.joint == null) continue;
                binding.restRotation = binding.joint.localRotation;
                // Start with no reading accepted.  A value serialized from an
                // earlier run would otherwise let the deadband swallow the first
                // packet and hold the joint on a stale target.
                binding.targetRotation = binding.restRotation;
                binding.appliedAngle = float.NaN;
            }
        }

        private void BuildLookup()
        {
            _byName.Clear();
            foreach (var binding in joints)
            {
                if (!string.IsNullOrEmpty(binding.axisName))
                {
                    _byName[binding.axisName] = binding;
                }
            }
        }

        void Update()
        {
            if (receiver == null) return;

            ApplyOrientationTint(receiver.Orientation);
            UpdateStableOrientation(receiver.Orientation);

            bool stale = receiver.SecondsSinceLastPacket > staleAfterSeconds;
            float blend = smoothing <= 0f
                ? 1f
                : 1f - Mathf.Exp(-Time.deltaTime / smoothing);

            foreach (var binding in joints)
            {
                if (binding.joint == null) continue;

                Quaternion target = binding.restRotation;
                float angle;

                // The wrist can follow the recognised orientation instead of
                // the raw axis.  Pronation is what shows the palm or the back of
                // the hand, so mapping front/side/back onto a 0..180 turn is
                // both what the operator sees and much steadier than the
                // measured wrist_rot.
                if (wristSource == WristSource.Orientation &&
                    binding.axisName == "wrist_rot")
                {
                    float wristDegrees = OrientationWristDegrees(_stableOrientation);
                    if (wristDegrees < 0f) continue;   // unknown: hold the pose

                    Quaternion wristTarget = binding.restRotation *
                        Quaternion.AngleAxis(wristDegrees, binding.rotationAxis);
                    if (Quaternion.Angle(binding.joint.localRotation, wristTarget) < 0.05f)
                        continue;
                    binding.joint.localRotation = Quaternion.Slerp(
                        binding.joint.localRotation, wristTarget, blend);
                    continue;
                }

                // A coupled joint has no channel of its own: it reads another
                // axis (the DIP follows the PIP) and applies a fraction of it.
                bool coupled = !string.IsNullOrEmpty(binding.coupleTo);
                string sourceAxis = coupled ? binding.coupleTo : binding.axisName;
                float scale = coupled ? binding.coupleFactor : 1f;

                if (!stale && receiver.TryGetAngle(sourceAxis, out angle))
                {
                    // Deadband on the INPUT: a pose being held repeats the same
                    // angle every packet, but the value still carries a little
                    // float noise.  Feeding that noise into the joint makes the
                    // hand shimmer while it is supposed to be still.
                    //
                    // The noise is ignored, but the joint keeps converging on
                    // the last accepted target.  Holding it where it stood
                    // instead stopped it after one blend step per new angle --
                    // about a quarter of the way at 60 fps with a 0.05 s blend --
                    // so a closed fist stayed visibly short of closed and every
                    // movement advanced in steps.
                    bool unchanged =
                        !float.IsNaN(binding.appliedAngle) &&
                        Mathf.Abs(angle - binding.appliedAngle) <= angleDeadband;

                    if (!unchanged)
                    {
                        binding.appliedAngle = angle;
                        binding.targetRotation = binding.restRotation * Quaternion.AngleAxis(
                            FlexionDegrees(binding, angle) * scale * flexionGain,
                            binding.rotationAxis);
                    }
                    target = binding.targetRotation;
                }
                else if (!relaxWhenStale)
                {
                    continue;   // hold whatever pose we are in
                }

                // Deadband on the OUTPUT: once the joint has settled on the
                // target, stop writing to it.  Slerping every frame toward an
                // almost-identical rotation is the other source of shimmer.
                if (Quaternion.Angle(binding.joint.localRotation, target) < 0.05f)
                {
                    continue;
                }

                binding.joint.localRotation = Quaternion.Slerp(
                    binding.joint.localRotation, target, blend);
            }
        }

        /// <summary>
        /// Wrist angle for a recognised orientation, or -1 when unknown.
        ///
        /// Returning a sentinel rather than a default keeps "we do not know"
        /// from snapping the hand to some arbitrary angle mid-gesture.
        /// </summary>
        /// Debounce the orientation before the wrist is allowed to follow it.
        ///
        /// "unknown" (edge-on) never counts as a change: the detector is least
        /// certain there, so the last confident answer is held rather than
        /// letting an ambiguous frame drag the wrist back to neutral.
        private void UpdateStableOrientation(string orientation)
        {
            if (string.IsNullOrEmpty(orientation) || orientation == "unknown") return;

            if (orientation == _stableOrientation)
            {
                _pendingOrientation = null;
                _pendingOrientationFrames = 0;
                return;
            }

            if (orientation != _pendingOrientation)
            {
                _pendingOrientation = orientation;
                _pendingOrientationFrames = 0;
            }

            _pendingOrientationFrames++;
            if (_pendingOrientationFrames >= Mathf.Max(1, wristHoldFrames))
            {
                _stableOrientation = orientation;
                _pendingOrientation = null;
                _pendingOrientationFrames = 0;
            }
        }

        private float OrientationWristDegrees(string orientation)
        {
            switch (orientation)
            {
                case "front": return frontWristDegrees;
                case "side": return sideWristDegrees;
                case "back": return backWristDegrees;
                default: return -1f;
            }
        }

        /// <summary>
        /// Colour the hand by the recognised orientation.
        ///
        /// "Which side am I looking at" is genuinely hard to judge on a grey
        /// model, and a subtly inverted front/back call is easy to miss.  A
        /// colour makes the answer immediate: show your palm, the hand should
        /// turn green; if it turns blue, the detector is inverted.
        /// </summary>
        private void ApplyOrientationTint(string orientation)
        {
            if (!tintByOrientation || _renderers == null || _renderers.Length == 0) return;
            if (orientation == _lastOrientation) return;
            _lastOrientation = orientation;

            Color tint;
            switch (orientation)
            {
                case "front": tint = frontColor; break;
                case "back": tint = backColor; break;
                case "side": tint = sideColor; break;
                default: tint = unknownColor; break;
            }

            foreach (var renderer in _renderers)
            {
                if (renderer == null) continue;
                renderer.GetPropertyBlock(_tintBlock);
                _tintBlock.SetColor("_Color", tint);
                renderer.SetPropertyBlock(_tintBlock);
            }
        }

        /// Map a stream angle onto this joint's rotation, in degrees.
        ///
        /// Uses the global calibration when enabled, so re-calibrating for a
        /// different hand is one edit instead of fifteen.
        public float FlexionDegrees(JointBinding binding, float streamAngle)
        {
            // The wrist is not a flexion joint: 90 is neutral and the value is a
            // signed offset (pronation one way, supination the other).
            if (binding.centeredOnNeutral)
            {
                float offset = streamAngle - 90f;
                return binding.invert ? -offset : offset;
            }

            float extended = useGlobalCalibration
                ? globalExtendedAngle
                : binding.extendedAngle;
            float flexed = useGlobalCalibration
                ? globalFlexedAngle
                : binding.flexedAngle;

            // InverseLerp handles a flexedAngle below extendedAngle, which is
            // the normal case (124 < 179) and the opposite of what Lerp expects.
            float fold = Mathf.Clamp01(
                Mathf.InverseLerp(extended, flexed, streamAngle));
            float degrees = fold * binding.maxFlexionDegrees;
            return binding.invert ? -degrees : degrees;
        }
    }
}
