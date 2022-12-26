import argparse

def rcs_interface(subparsers):
    parser_tool = subparsers.add_parser(
        "rcstool",
        help="tools used in rcs-magus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #For reconstructions, get a slab
    parser_tool.add_argument(
        "--getslab",
        "-g",
        action="store_true",
        help="get the slab model used in rcs-magus",
    )
    parser_tool.add_argument(
        "-f",
        "--filename",
        type=str,
        default= '',
        help="defaults is './Ref/layerslices.traj' of slab model and 'results' for analyze results."
    )
    parser_tool.add_argument(
        "-s",
        "--slabfile",
        type=str,
        default= 'slab.vasp',
        help="slab file",
    )

    #generation energy analizer, a quick version of summary
    parser_tool.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="get energy tendency of evolution",
    )
    
    parser_tool.add_argument(
        "-e",
        "--to-excel",
        type=str,
        default= None,
        help="output to excel",
    )
    parser_tool.add_argument(
        "-p",
        "--to_plt",
        type=str,
        default= None,
        help="output to plot",
    )

    return 