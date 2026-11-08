// PoseReceiver.cs
//
// Listens for joint-angle packets from the Python bridge and holds the most
// recent pose for the rest of the scene to read.
//
// Design notes
// ------------
// * The socket runs on a background thread, so nothing here may touch the Unity
//   API.  Packets are parsed into a plain struct and published under a lock;
//   HandRig reads that struct from Update() on the main thread.
//
// * UDP is connectionless: "no client sending" is not an error and cannot be
//   detected by connecting.  The receiver simply waits, and reports staleness
//   via SecondsSinceLastPacket so the rig can decide what to do about it.
//
// * The angle array's ORDER is part of the wire format.  See the Python side
//   (src/ui/unity_bridge.py, AXIS_ORDER).  The hello packet carries the names;
//   this script logs them so a mismatch is visible in the console rather than
//   silently driving the wrong fingers.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace BionicHand
{
    /// One decoded pose packet.
    public struct PoseSample
    {
        public float[] angles;      // in AXIS_ORDER, degrees
        public string gesture;      // may be null
        public string motion;       // dynamic gesture, may be null
        public string orientation;  // may be null
        public float orientationConfidence;
        public float receivedAt;    // Time.realtimeSinceStartup when it arrived
        public bool valid;
    }

    [Serializable]
    internal class PosePacketJson
    {
        public float t;
        public float[] a;
        public string g;
        public string m;
        public string o;
        public float oc;
    }

    [Serializable]
    internal class HelloPacketJson
    {
        public bool hello;
        public string[] n;
    }

    public class PoseReceiver : MonoBehaviour
    {
        [Header("Network")]
        [Tooltip("Port the Python bridge sends to. Must match unity_bridge.DEFAULT_PORT.")]
        public int port = 8888;

        [Tooltip("Log every received axis name list once, on the first hello packet.")]
        public bool logAxisOrder = true;

        /// Joint angles of the most recent packet, in AXIS_ORDER. Empty until
        /// the first packet arrives.
        public float[] Angles
        {
            get { lock (_lock) { return _latest.valid ? _latest.angles : null; } }
        }

        public string Gesture
        {
            get { lock (_lock) { return _latest.gesture; } }
        }

        /// Dynamic gesture ("clockwise", "move", ...), or null.
        public string Motion
        {
            get { lock (_lock) { return _latest.motion; } }
        }

        public string Orientation
        {
            get { lock (_lock) { return _latest.orientation; } }
        }

        public float OrientationConfidence
        {
            get { lock (_lock) { return _latest.orientationConfidence; } }
        }

        /// Seconds since the last packet, or a large number if none ever came.
        public float SecondsSinceLastPacket
        {
            get
            {
                lock (_lock)
                {
                    if (!_latest.valid) return float.MaxValue;
                    return Time.realtimeSinceStartup - _latest.receivedAt;
                }
            }
        }

        public bool HasData
        {
            get { lock (_lock) { return _latest.valid; } }
        }

        /// Names announced by the bridge's hello packet, so the rig can map
        /// angles by name instead of by hard-coded index.
        public IReadOnlyList<string> AxisNames
        {
            get { lock (_lock) { return _axisNames; } }
        }

        private readonly object _lock = new object();
        private PoseSample _latest;
        private List<string> _axisNames = new List<string>();

        /// Canonical axis order, mirroring AXIS_ORDER in
        /// src/ui/unity_bridge.py.  Used as a fallback when the hello packet was
        /// missed.
        ///
        /// The hello packet is sent once, when the Python stream starts.  A
        /// Unity session that began playing afterwards never sees it, leaves
        /// _axisNames empty, and then every angle lookup misses -- the pose
        /// arrives, the HUD says "live", and nothing moves.  That is a silent
        /// failure, so the order is duplicated here as a safety net.
        private static readonly List<string> CanonicalOrder = new List<string>
        {
            "thumb_cmc", "thumb_mcp", "thumb_ip",
            "index_mcp", "index_pip",
            "middle_mcp", "middle_pip",
            "ring_mcp", "ring_pip",
            "pinky_mcp", "pinky_pip",
            "wrist_rot",
        };

        private UdpClient _client;
        private Thread _thread;
        private volatile bool _running;

        void OnEnable()
        {
            try
            {
                // ReuseAddress lets a restarted editor rebind immediately
                // instead of failing with "address already in use".
                _client = new UdpClient();
                _client.Client.SetSocketOption(
                    SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                _client.Client.Bind(new IPEndPoint(IPAddress.Loopback, port));
                _running = true;
                _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "PoseReceiver" };
                _thread.Start();
                Debug.Log($"[PoseReceiver] listening on udp 127.0.0.1:{port}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[PoseReceiver] could not bind port {port}: {e.Message}");
                _running = false;
            }
        }

        void OnDisable()
        {
            _running = false;
            try { _client?.Close(); } catch { /* shutting down anyway */ }
            _client = null;
            _thread = null;
        }

        private void ReceiveLoop()
        {
            var remote = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    byte[] data = _client.Receive(ref remote);
                    HandlePacket(Encoding.UTF8.GetString(data));
                }
                catch (SocketException)
                {
                    // Fires when the socket is closed during shutdown, and on
                    // Windows when a previous send got an ICMP port-unreachable.
                    // Neither is fatal: keep listening.
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[PoseReceiver] bad packet: {e.Message}");
                }
            }
        }

        private void HandlePacket(string json)
        {
            // Hello and pose packets have different shapes and JsonUtility has
            // no way to discriminate, so check for the marker field directly.
            if (json.Contains("\"hello\""))
            {
                var hello = JsonUtility.FromJson<HelloPacketJson>(json);
                if (hello != null && hello.n != null)
                {
                    lock (_lock) { _axisNames = new List<string>(hello.n); }
                    if (logAxisOrder)
                    {
                        Debug.Log("[PoseReceiver] axis order: " + string.Join(", ", hello.n));
                    }
                }
                return;
            }

            var packet = JsonUtility.FromJson<PosePacketJson>(json);
            if (packet == null || packet.a == null) return;

            var sample = new PoseSample
            {
                angles = packet.a,
                gesture = string.IsNullOrEmpty(packet.g) ? null : packet.g,
                motion = string.IsNullOrEmpty(packet.m) ? null : packet.m,
                orientation = string.IsNullOrEmpty(packet.o) ? null : packet.o,
                orientationConfidence = packet.oc,
                // RealtimeSinceStartup is main-thread-ish, but reading it here
                // would still be a Unity API call off the main thread. Stamp it
                // on the main thread instead (see Update).
                receivedAt = 0f,
                valid = true,
            };

            lock (_lock)
            {
                _latest = sample;
                _pendingStamp = true;
            }
        }

        private bool _pendingStamp;

        void Update()
        {
            // Stamp arrival time on the main thread so SecondsSinceLastPacket
            // uses a clock Unity is happy with.
            if (_pendingStamp)
            {
                lock (_lock)
                {
                    _latest.receivedAt = Time.realtimeSinceStartup;
                    _pendingStamp = false;
                }
            }
        }

        /// Look up an axis angle by name. Returns false if unknown or no data.
        public bool TryGetAngle(string axisName, out float degrees)
        {
            degrees = 90f;
            lock (_lock)
            {
                if (!_latest.valid) return false;

                // Fall back to the canonical order when the hello packet was
                // missed (Unity started after the Python stream).  Without this
                // the pose arrives and is silently ignored.
                var names = _axisNames.Count > 0 ? _axisNames : CanonicalOrder;

                int index = names.IndexOf(axisName);
                if (index < 0 || index >= _latest.angles.Length) return false;
                degrees = _latest.angles[index];
                return true;
            }
        }
    }
}
