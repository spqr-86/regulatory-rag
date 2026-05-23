import json

try:
    with open(
        "/Users/petrbaldaev/.local/share/opencode/tool-output/tool_c3ea21fd7001jrVnWfS27CuRXt",
        "r",
    ) as f:
        data = json.load(f)

    version = "2.60.0"
    if version in data.get("releases", {}):
        dist = data["releases"][version][
            0
        ]  # assuming first wheel/sdist has metadata? No, PyPI JSON puts requires_dist in 'info' usually for latest, but releases might not have it unless it's full metadata.
        # Wait, 'info' key has metadata for the *latest* release. Let's check 'info' version.
        info_version = data["info"]["version"]
        print(f"Info version: {info_version}")

        if info_version == version:
            print("Dependencies for", version)
            for req in data["info"]["requires_dist"] or []:
                print(req)
        else:
            print(
                f"Info version is {info_version}, not {version}. Checking releases..."
            )
            # Often releases don't have full metadata in the JSON endpoint, only url/digest.
            # But let's check.
            pass

except Exception as e:
    print(f"Error: {e}")
