import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    t_y = 10
    t_x = 4


    # Create a matrix of zeros
    matrix = np.zeros((t_x, t_y))

    # Fill the matrix with the indices from the for loop
    for y in range(t_y):
        for x in range(max(0, t_x - (t_y - y)), min(t_x, y + 1)):
            matrix[x, y] = 1

    # Plot the matrix
    plt.imshow(matrix, cmap='gray')
    plt.title('Matrix with Indices')
    plt.xlabel('y')
    plt.xticks(np.arange(t_y))
    plt.yticks(np.arange(t_x))
    plt.ylabel('x')
    plt.colorbar()

    # Save the image
    plt.savefig('/home/dsi/moradim/SpeechRepainting/glow-tts/matrix_image_2.png')
    plt.show()
    # print(f"x={x}, y={y}")
# print("move frame")

    # # Detect indices of target elements
    # phoneme_with_silence = [10, 2, 7, 1, 8, 11, 1]
    # durations_with_silence =  [4, 7, 2, 10, 2, 3, 8]
    # phoneme_without_silence = [10, 2, 7, 8, 11]
    # durations_without_silence = [4, 7, 2, 2, 3]
    # interspersed_phoneme_sequence = intersperse(phoneme_without_silence, 1)
    # interspersed_phoneme_duration = intersperse(durations_without_silence, 1)
    
    # indices = np.where(np.isin(phoneme_with_silence, 1))[0]
    # sort_indices = np.sort(indices)
    # sort_indices_without_silence = sort_indices - np.arange(len(sort_indices))
    # for i in range(len(sort_indices_without_silence)):
    #     interspersed_phoneme_duration[sort_indices_without_silence[i] * 2] = durations_with_silence[sort_indices[i]]

    # print(interspersed_phoneme_duration)

