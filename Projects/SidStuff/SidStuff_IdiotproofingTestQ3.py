# Q3 - An ordered list of strings
import time

def find_longest_sublist(strings):

    if not strings:
        return []
    
    current_sequence = [strings[0]]
    longest_sequence = [strings[0]]

    for i in range(1, len(strings)):
        if len(strings[i]) > len(current_sequence[-1]):
            current_sequence.append(strings[i])
        else:
            current_sequence = [strings[i]]

        if len(current_sequence) > len(longest_sequence):
            longest_sequence = current_sequence.copy()
        
    return longest_sequence

def test_sequence_finder():
    #Test case 1
    test1 = ['one', 'two', 'three', 'four', 'five', 'six']
    result1 = find_longest_sublist(test1)
    print("\n Test 1 (from question):")
    print(f"Input:{test1}")
    print(f"result: {result1}")
    print(f"Lengths: {[len(s) for s in result1]}")

    #Test case 2
    test2 = ['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'dddd']
    result2 = find_longest_sublist(test2)
    print("\n Test 2 (varying lengths):")
    print(f"Input:{test2}")
    print(f"result: {result2}")
    print(f"Lengths: {[len(s) for s in result2]}")

    #Test case 3
    test3 = ['fifteen', 'twelve', 'seven', 'three', 'two', 'one']
    result3 = find_longest_sublist(test3)
    print("\n Test 3 (decending size):")
    print(f"Input:{test3}")
    print(f"result: {result3}")
    print(f"Lengths: {[len(s) for s in result3]}")

    #Test case 4
    test4 = ['cat', 'fat', 'mat', 'dog', 'fog', 'sid']
    result4 = find_longest_sublist(test4)
    print("\n Test 4 (same length):")
    print(f"Input:{test4}")
    print(f"result: {result4}")
    print(f"Lengths: {[len(s) for s in result4]}")

    print("\nPerformance test:")
    large_input = ['a' * i for i in range(1,10001)]
    start = time.time()
    result_large = find_longest_sublist(large_input)
    end = time.time()
    print(f"Time taken for 1000 strings: {end - start:.4f} seconds")
    print(f"Length of result: {len(result_large)}")

if __name__ == "__main__":
    test_sequence_finder()



