import json

try:
    with open(
        "/Users/petrbaldaev/.local/share/opencode/tool-output/tool_c3ea2d18e001UMviIdvg8D8Uy5",
        "r",
    ) as f:
        data = json.load(f)

    version = data["info"]["version"]
    print(f"Latest version: {version}")

    print(f"Dependencies for {version}:")
    for req in data["info"]["requires_dist"] or []:
        print(req)

except Exception as e:
    print(f"Error: {e}")
