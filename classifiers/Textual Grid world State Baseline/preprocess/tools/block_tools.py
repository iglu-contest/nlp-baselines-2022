from collections import Counter
import inflect

WordToDigits = inflect.engine()


VOXELWORLD_GROUND_LEVEL = 63

block_colour_map = {
    # voxelworld's colour id : iglu colour id
    0: 0,  # air
    57: 1,  # blue
    50: 6,  # yellow
    59: 2,  # green
    47: 4,  # orange
    56: 5,  # purple
    60: 3,  # red
}

block_colour_name_map = {
    # voxelworld's colour id : iglu colour id
    0: "air",
    1: "blue",
    6: "yellow",
    2: "green",
    4: "orange",
    5: "purple",
    3: "red",
}


def fix_xyz(x, y, z):
    XMAX = 11
    YMAX = 9
    ZMAX = 11
    COORD_SHIFT = [5, -63, 5]

    x += COORD_SHIFT[0]
    y += COORD_SHIFT[1]
    z += COORD_SHIFT[2]

    index = z + y * YMAX + x * YMAX * ZMAX
    new_x = index // (YMAX * ZMAX)
    index %= YMAX * ZMAX
    new_y = index // ZMAX
    index %= ZMAX
    new_z = index % ZMAX

    new_x -= COORD_SHIFT[0]
    new_y -= COORD_SHIFT[1]
    new_z -= COORD_SHIFT[2]

    return new_x, new_y, new_z


def transform_block(block):
    """Adjust block coordinates and replace id."""
    x, y, z, bid = block
    y = y - VOXELWORLD_GROUND_LEVEL - 1
    bid = block_colour_map.get(bid, 5)  # TODO: some blocks have id 1, check why
    return x, y, z, bid


def count_block_colors(blocks: list) -> dict:

    colours = [block_colour_name_map[block[3]] for block in blocks]

    colour_freqs = Counter(colours)

    return colour_freqs


def create_context_colour_count(colour_freq: dict) -> list:

    res = [f"There are {sum(colour_freq.values())} blocks in total."]
    res.append(f"There are {len(colour_freq)} different colours.")
    for colour in colour_freq:

        if colour != "air":

            res.append(f"There are {colour_freq[colour]} {colour} blocks ")

    return "".join(res)


def get_color_counter_by_level(blocks: list):

    blocks.copy()
    blocks.sort(key=lambda block: block[1])

    colorvec_by_level = []
    level = -1
    h = -100
    # accomodate blocks by level
    for block in blocks:
        # print(level)
        if block[1] != h:
            level += 1
            colorvec_by_level.append([block])
            h = block[1]
        else:
            colorvec_by_level[level].append(block)
    # apply the counter on each level
    for i in range(len(colorvec_by_level)):

        colorvec_by_level[i] = count_block_colors(colorvec_by_level[i])

    return colorvec_by_level


def aux_num_colour_context(colour_freq):

    colourstrs = [f"{colour_freq[colour]} {colour}" for colour in colour_freq]

    return " and ".join(colourstrs)


def create_context_colour_by_height_level(colour_freq_by_level: list):
    """Take a list from downwards to upwards of the number of blocks in each level

    In the first level there are. Above there are.....At the top there are.
    In the case first is the top say so.


    """

    res = [f"There are {len(colour_freq_by_level)} levels."]

    # maybe add some meadssure of how many block by height
    res.append(
        f"There are {sum([sum(tabl.values()) for tabl in colour_freq_by_level])} different blocks."
    )

    for i in range(len(colour_freq_by_level)):
        colour_freq = colour_freq_by_level[i]
        # if colour != "air":
        # For each ordinal level it says how many blocks of each colour there are
        res.append(
            f"""{ 'Above at' if i>0 else 'At'} the {WordToDigits.ordinal(i) } level there are {aux_num_colour_context(colour_freq)}  blocks """
        )

    return " ".join(res)


# Maybe better to put just above there are.....
