import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    t_y = 21
    t_x = 9
    batch_bool = True
    if batch_bool:
        t_y = np.array([t_y, t_y-1, t_y-2, t_y-3])
        t_x = np.array([t_x, t_x-1, t_x-2, t_x-3])
        
        t_x = np.expand_dims(t_x, (1,2))
        t_y = np.expand_dims(t_y, (1,2))

    
    # Create a matrix of zeros
    # matrix = np.zeros((t_x, t_y))

    # Fill the matrix with the indices from the for loop
    # for y in range(t_y):
    #     for x in range(max(0, t_x - 2*(t_y - y)+1), min(t_x, 2*y+1)):
    #         matrix[x, y] = 1
            

    # Initialize the matrix with zeros
    if batch_bool:
        matrix = np.zeros((4, 20, 30))
        x_indices, y_indices = np.indices(matrix[0].shape)
        y_indices = np.expand_dims(y_indices, (0))
        x_indices = np.expand_dims(x_indices, (0))
    else:
        matrix = np.zeros((20, 30))
        x_indices, y_indices = np.indices(matrix.shape)
    # Use broadcasting to calculate the conditions for filling

    condition = (x_indices >= np.maximum(0, t_x - 2 * (t_y - y_indices) + 1)) & \
                (x_indices < np.minimum(t_x, 2 * y_indices + 1))


    # condition = (x_indices >= np.maximum(0, t_x - 2 * (t_y - y_indices) + 1)) & \
    #             (x_indices < np.minimum(t_x, 2 * y_indices + 1))
    # matrix = np.where(condition, 1, 0)
    # Set the values to 1 where the condition is met
    matrix[condition] = 1

    if batch_bool:
        for i in range(matrix.shape[0]):
            # Plot the matrix
            plt.imshow(matrix[i], cmap='gray')
            plt.title('Matrix with Indices')
            plt.xlabel('y')
            plt.xticks(np.arange(t_y[i, 0, 0]))
            plt.yticks(np.arange(t_x[i, 0, 0]))
            plt.ylabel('x')
            plt.colorbar()

            # Save the image
            plt.savefig(f'/home/dsi/moradim/SpeechRepainting/glow-tts/matrix_image_3_sample={i}.png')
            plt.show()
            plt.close()
    else:
        plt.imshow(matrix, cmap='gray')
        plt.title('Matrix with Indices')
        plt.xlabel('y')
        plt.xticks(np.arange(t_y))
        plt.yticks(np.arange(t_x))
        plt.ylabel('x')
        plt.colorbar()

        # Save the image
        plt.savefig(f'/home/dsi/moradim/SpeechRepainting/glow-tts/matrix_image_one_sample.png')
        plt.show()
        plt.close()
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

