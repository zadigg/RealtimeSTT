"""
Audio device utilities for dual-source transcription (microphone vs PC output).
"""
import platform
import logging

logger = logging.getLogger("realtimestt")


def find_loopback_device(pyaudio_instance, device_name_substring=None, host_api=0):
    """
    Find the loopback/system audio device index.
    
    On macOS: Looks for BlackHole (install via: brew install blackhole-2ch)
    On Windows: Looks for Stereo Mix (enable in Sound settings)
    
    Args:
        pyaudio_instance: PyAudio instance
        device_name_substring: Optional substring to match (e.g. "blackhole", "stereo mix")
        host_api: Host API index to filter by (0 = default)
    
    Returns:
        (device_index, devices_info_str) - index or None if not found
    """
    if device_name_substring is None:
        if platform.system() == "Darwin":
            device_name_substring = "blackhole"
        elif platform.system() == "Windows":
            device_name_substring = "stereo mix"
        else:
            device_name_substring = "loopback"  # Generic fallback
    
    devices_info = ""
    for i in range(pyaudio_instance.get_device_count()):
        dev = pyaudio_instance.get_device_info_by_index(i)
        devices_info += f"  {dev['index']}: {dev['name']} (hostApi: {dev['hostApi']})\n"
        
        if (device_name_substring.lower() in dev['name'].lower() and 
            dev.get('maxInputChannels', 0) > 0):
            if host_api is None or dev['hostApi'] == host_api:
                return dev['index'], devices_info
    
    return None, devices_info


def find_default_microphone(pyaudio_instance):
    """Get the default microphone/input device index."""
    try:
        default = pyaudio_instance.get_default_input_device_info()
        return default['index']
    except Exception as e:
        logger.warning(f"Could not get default input device: {e}")
        return None


def get_loopback_setup_instructions():
    """Return platform-specific instructions for setting up loopback capture."""
    if platform.system() == "Darwin":
        return """
To capture PC audio (YouTube, etc.) on macOS:
1. Install BlackHole:  brew install blackhole-2ch
2. Open Audio MIDI Setup → Create Multi-Output Device
3. Select BlackHole 2ch + your speakers
4. Set Multi-Output Device as system output
5. Restart this script
"""
    elif platform.system() == "Windows":
        return """
To capture PC audio on Windows:
1. Right-click speaker icon → Sounds → Recording tab
2. Enable "Stereo Mix" (or "What U Hear") if available
3. Set as default device or ensure it's not disabled
4. Restart this script
"""
    else:
        return """
To capture PC audio: Install a loopback driver for your OS
(e.g. PulseAudio loopback on Linux, or similar).
"""
