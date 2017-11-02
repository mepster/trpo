#! /usr/bin/env python3

import argparse, os

def main(name):
    name=name.split("/")[-1] # remove preceding/path/components/to/name
    shortname="-".join(name.split("-")[:-1]) # remove -46620 episode number from end of name
    print(name, shortname)
    cmd="tar -czvf aws-saves.tgz saves/*/"+name+"* log-files/Osim/"+shortname
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('make tar file'))
    parser.add_argument('name', type=str, help='name of run e.g., Nov-01_20:05:27-46620. Preceding path components are automatically removed.')

    args = parser.parse_args()
    main(**vars(args))

