import argparse
import os
import subprocess


def ssh(pem_file, remote_address):
    command = f'ssh -i "{pem_file}" ubuntu@{remote_address}'
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSH into an EC2 instance")
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

    args = parser.parse_args()

    ssh(args.pem_file, args.remote_address)
