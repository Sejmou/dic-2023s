import argparse
import os
import subprocess


def copy_to_remote(pem_file, remote_address, source_path, destination_path):
    command = f'scp -i "{pem_file}" {"-r" if os.path.isdir(source_path) else ""} {source_path} {remote_address}:{destination_path}'
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files to an EC2 instance")
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
        "-s", "--source-path", required=True, help="Path to source directory"
    )
    parser.add_argument(
        "-d",
        "--destination-path",
        required=True,
        help="Path to destination directory",
    )

    args = parser.parse_args()

    copy_to_remote(
        args.pem_file,
        "ubuntu@" + args.remote_address,
        args.source_path,
        args.destination_path,
    )
