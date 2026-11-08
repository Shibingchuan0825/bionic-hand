// StatusHud.cs
//
// Shows the link state, the recognised gesture and the hand orientation on
// screen.  Deliberately plain IMGUI: it is diagnostic overlay, not part of the
// hand model, and it needs to work before any canvas or font asset exists.
//
// The orientation readout shows "unknown" as its own state rather than hiding
// it.  Edge-on is where the classifier is weakest (the palm's projected area
// collapses toward a line), so pretending to a confident answer there would be
// worse than admitting the ambiguity.

using UnityEngine;

namespace BionicHand
{
    public class StatusHud : MonoBehaviour
    {
        public PoseReceiver receiver;

        [Tooltip("Draw the overlay at all.")]
        public bool visible = true;

        [Tooltip("Seconds with no packet before the link is called stale.")]
        public float staleAfterSeconds = 1.0f;

        private GUIStyle _style;
        private GUIStyle _warn;

        void Awake()
        {
            if (receiver == null) receiver = FindObjectOfType<PoseReceiver>();
        }

        void OnGUI()
        {
            if (!visible || receiver == null) return;

            if (_style == null)
            {
                _style = new GUIStyle(GUI.skin.label) { fontSize = 16 };
                _style.normal.textColor = Color.white;
                _warn = new GUIStyle(_style);
                _warn.normal.textColor = new Color(1f, 0.6f, 0.2f);
            }

            float age = receiver.SecondsSinceLastPacket;
            bool live = age <= staleAfterSeconds;

            var box = new Rect(12, 12, 320, 140);
            GUI.Box(box, GUIContent.none);

            float y = box.y + 8;
            var line = new Rect(box.x + 10, y, box.width - 20, 22);

            GUI.Label(line, live
                ? $"Pose stream: live ({age * 1000f:F0} ms)"
                : "Pose stream: no data", live ? _style : _warn);
            y += 24;

            line.y = y;
            GUI.Label(line, $"Gesture: {receiver.Gesture ?? "-"}", _style);
            y += 24;

            line.y = y;
            string orientation = DescribeOrientation(receiver.Orientation,
                                                     receiver.OrientationConfidence);
            GUI.Label(line, $"Orientation: {orientation}", _style);
            y += 24;

            line.y = y;
            GUI.Label(line, $"Joints received: {(receiver.Angles?.Length ?? 0)}", _style);
        }

        /// Render the orientation string, marking low confidence explicitly.
        static string DescribeOrientation(string orientation, float confidence)
        {
            if (string.IsNullOrEmpty(orientation)) return "-";
            if (orientation == "unknown") return "unknown (edge-on)";

            string label;
            switch (orientation)
            {
                case "front": label = "palm toward camera"; break;
                case "back":  label = "back of hand toward camera"; break;
                case "side":  label = "edge-on"; break;
                default:      label = orientation; break;
            }
            // Below 0.7 the reading is close to the edge-on singularity; say so
            // rather than presenting a coin-flip as a fact.
            return confidence > 0f && confidence < 0.7f
                ? $"{label} (low confidence {confidence:F2})"
                : label;
        }
    }
}
