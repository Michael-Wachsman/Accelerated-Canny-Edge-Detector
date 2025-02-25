def count_differences(file1, file2):
    try:
        # Read and split the data from both files
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            data1 = f1.read().split()
            data2 = f2.read().split()
        
        # Ensure both files have the same number of values
        if len(data1) != len(data2):
            raise ValueError("Files do not have the same number of values. Cannot compare.")
        
        # Convert data to integers
        data1 = list(map(int, data1))
        data2 = list(map(int, data2))
        
        # Count the number of differing values
        diff_count = sum(1 for a, b in zip(data1, data2) if a != b)
        percent_diff = diff_count*100 / len(data1)
        print(f"Number of differing pixels: {diff_count}")
        print(f"Percentage of differing pixels: {percent_diff}%")
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python count_differences.py <file1> <file2>")
        sys.exit(1)

    # Get file paths from command-line arguments
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Count differences
    count_differences(file1, file2)
        