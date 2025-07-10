import subprocess
import json

def run_maas_command(command):
    """Runs a MAAS CLI command and returns parsed JSON output."""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing MAAS CLI command:", e.stderr)
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON output.")
        return None

def get_maas_machines(profile="admin"):
    """Get a list of machines from MAAS."""
    cmd = ["maas", profile, "machines", "read"]
    return run_maas_command(cmd)

def get_maas_boot_resources(profile="admin"):
    """Get boot resources from MAAS."""
    cmd = ["maas", profile, "boot-resources", "read"]
    return run_maas_command(cmd)

# --- Usage Example ---
if __name__ == "__main__":
    maas_profile = "admin"  # change to your MAAS profile name if different

    print("Fetching MAAS Machines...")
    machines = get_maas_machines(maas_profile)
    if machines is not None:
        print(json.dumps(machines, indent=2))

    print("\nFetching Boot Resources...")
    boot_resources = get_maas_boot_resources(maas_profile)
    if boot_resources is not None:
        print(json.dumps(boot_resources, indent=2))
