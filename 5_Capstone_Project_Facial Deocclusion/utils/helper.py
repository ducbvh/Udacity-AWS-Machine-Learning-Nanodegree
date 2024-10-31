import os
import matplotlib.pyplot as plt


def generate_images(model, test_input, tar, save_folder="generated_images", epoch=0, show_plt=False):
    # Ensure the save folder exists
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Generate predictions
    prediction = model(test_input, training=True)
    
    # Create a new figure
    plt.figure(figsize=(15, 15))

    # Prepare images and titles for display
    display_list = [test_input[0] * 0.5 + 0.5, tar[0] * 0.5 + 0.5, prediction[0] * 0.5 + 0.5]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    
    # Save the figure to the specified folder and filename
    filename=f"output_image_{epoch}.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    if show_plt:
        plt.show()

    plt.close()  # Close the figure to free memory
    print(f"Image test results saved to {save_path}")


def plot_history_train(history_train, history_val, path_save, file_name="", total_history=True):
    plt.plot(history_train)
    plt.plot(history_val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if total_history:
        filename="loss_total_curve.png"
    else:
        filename=f"loss_{file_name}_curve.png"
    save_path = os.path.join(path_save, filename)
    plt.savefig(save_path)
    plt.close()