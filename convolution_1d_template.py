import numpy as np

def Convolution_1D(
    array1: np.array, array2: np.array, padding: str, stride: int = 1
) -> np.array:
    """
    Compute the convolution array1 * array2.
    
    Args:
    array1: np.array, input array
    array2: np.array, kernel
    padding: str, padding type ('full' or 'valid')
    stride: int, specifies how much we move the kernel at each step
    
    Carefully look at the formula for convolution in the problem statement, specifically the g[n-m] term.
    What does it indicate?
    Also note the constraints on sizes:
    - For padding='full', the sizes of array1 and array2 can be anything
    - For padding='valid', the size of array1 must be greater than or equal to the size of array2.
    
    Returns:
    np.array, output of the convolution
    """
    
    if len(array1) < len(array2):
        raise ValueError("For 'valid' padding, the size of array1 must be greater than or equal to the size of array2.")

 
    if padding == 'full':
        pad_width = array2.size - 1
        array1 = np.concatenate((np.zeros(pad_width), array1, np.zeros(pad_width)))
    elif padding != 'valid':
        raise ValueError("Padding must be 'full' or 'valid'.")
    
    output_size = (len(array1) - len(array2)) // stride + 1
    conv = np.zeros(output_size)

    for i in range(0, len(array1) - len(array2) + 1, stride):
        conv[i // stride] = np.sum(array1[i : i + len(array2)] * array2)

    return conv



def probability_sum_of_faces(p_A: np.array, p_B:np.array) -> np.array:
    """
    Compute the probability of the sum of faces of two unfair dice rolled together.

    Args:
        p_A (np.array): Probabilities of the faces of die A, from 1 to number of faces of die A
        p_B (np.array): Probabilities of the faces of die B, from 1 to number of faces of die B

    Returns:
        np.array: Probabilities of the sum of faces of die A and die B.
    
    Note that the sum of the faces cannot be 1, and hence the probability array should start from 2.
    For example, given two fair dice with 2 faces each,
    p_A = [0.5, 0.5] and p_B = [0.5, 0.5], the probabilities of sum of faces are

    Sum = 1 => Probability = 0
    Sum = 2 => Probability = 0.25 (1_A, 1_B)
    Sum = 3 => Probability = 0.5 (1_A, 2_B), (2_A, 1_B)
    Sum = 4 => Probability = 0.25 (2_A, 2_B)

    The output should start with the probability of 2, and hence the expected output is [0.25, 0.5, 0.25].
    """
    if p_A.size >= p_B.size :
        return Convolution_1D(p_A,np.flip(p_B),"full")
    else:
        return Convolution_1D(p_B,np.flip(p_A),"full")


''' test 
a1 = np.array([1,2,3,4,5])
a2 = np.array([1,2,1])

print(Convolution_1D(a1,a2,padding="full",stride=2))
'''
p_B = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
p_A = np.array([0.3, 0.4, 0.3])
x= probability_sum_of_faces(p_A,p_B)
print(x)