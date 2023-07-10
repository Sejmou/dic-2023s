import argparse
import os
import subprocess


def copy_files(pem_file, remote_address, source_path, destination_path):
    command = (
        f'scp -i "{pem_file}" -r {source_path} {remote_address}:{destination_path}'
    )
    subprocess.run(command, shell=True, check=True)


def main(pem_file, remote_address, cpu_only, local_models_path):
    config_source = "config"
    config_dest = "config"

    if cpu_only:
        script_source = "start_cpu.sh"
    else:
        script_source = "start_gpu.sh"

    script_dest = "start.sh"

    copy_files(pem_file, remote_address, config_source, config_dest)
    copy_files(pem_file, remote_address, script_source, script_dest)

    if local_models_path:
        if os.path.exists(local_models_path):
            copy_files(pem_file, remote_address, local_models_path, "models")
        else:
            raise ValueError("Invalid path to local models directory")
    else:
        print(
            f"No path to local pretrained models provided. Copying script for fetching models instead."
        )
        copy_files(pem_file, remote_address, "model_fetching", "model_fetching")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy files required for running Object Detection API to remote Ubuntu EC2 instance"
    )
    parser.add_argument(
        "-p",
        "--pem-file",
        default="/users/sejmou/.ssh/aws-sejmou.pem",
        help="Path to the PEM file",
    )
    parser.add_argument(
        "-r",
        "--remote-address",
        required=True,
        help="Public IPv4 DNS address of EC2 instance",
    )
    parser.add_argument(
        "-c", "--cpu-only", action="store_true", help="Flag for CPU-only"
    )
    parser.add_argument(
        "-l", "--local-models-path", help="Path to local models directory"
    )

    args = parser.parse_args()

    main(
        args.pem_file,
        "ubuntu@" + args.remote_address,
        args.cpu_only,
        args.local_models_path,
    )
