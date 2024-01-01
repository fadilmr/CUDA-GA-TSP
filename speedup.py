def calculate_speedup(single_block_time, multi_block_time):
    """
    Function to calculate the speedup of a program.
    :param single_block_time: Time taken to run the program with a single block.
    :param multi_block_time: Time taken to run the program with multiple blocks.
    :return: Speedup
    """
    try:
        speedup = single_block_time / multi_block_time
        return speedup
    except ZeroDivisionError:
        return "Error: Division by zero. The multi_block_time must be greater than zero."

# Test the function

print(calculate_speedup(6922.94, 21.1387))  # Output: 5.0