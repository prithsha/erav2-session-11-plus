
import math
import torch
import matplotlib.pyplot as plt

def convert_image_tensor_to_numpy_hwc_format(image_data):
    # Image format is CWH (Channel, hight and width)
    # Converting tensor to numpy to HWC (Height, Width , Channel) format
    return image_data.numpy().transpose((1, 2, 0))

def show_processed_images(selected_images, image_classes, rows = 4, cols = 5, fig_size=(10,4)):
    figure = plt.figure(figsize=fig_size)

    max_count = min(cols * rows, len(selected_images))
    for i in range(1, max_count + 1):
        img, label = selected_images[i-1]
        figure.add_subplot(rows, cols, i)        
        plt.title(image_classes[label])
        plt.axis("off")
        plt.imshow(img)
    plt.show()

def show_images_from_tensor_array(images_data, image_classes,  shape=(2,10)):

    rows = shape[0]
    cols = shape[1]
    processed_images_data = []
    max_count = min(cols * rows, len(images_data))
    for i in range(max_count): 
        image_data, label = images_data[i]
        if(isinstance(image_data, torch.Tensor)):
            image_data = image_data.cpu()
        if(isinstance(label, torch.Tensor)):
            label = label.cpu()
        image_data = convert_image_tensor_to_numpy_hwc_format(image_data)
        processed_images_data.append((image_data, label))

    show_processed_images(processed_images_data,image_classes,rows, cols)

def randomly_show_images_from_tensor_array(images_data, image_classes, shape=(2,10), fig_size=(10,4)):

    rows = shape[0]
    cols = shape[1]
    processed_images_data = []
    max_count = min(cols * rows, len(images_data))
    for _ in range(max_count):
        # Generate a random integer within dataset length
        sample_idx = torch.randint(0, len(images_data), size=(1,))  
        image_data, label = images_data[sample_idx.item()]
        if(isinstance(image_data, torch.Tensor)):
            image_data = image_data.cpu()
        if(isinstance(label, torch.Tensor)):
            label = label.cpu()

        image_data = convert_image_tensor_to_numpy_hwc_format(image_data)
        processed_images_data.append((image_data, label))

    show_processed_images(processed_images_data,image_classes,rows, cols, fig_size)

def show_image(image_data, label, fig_size=(3,3)):
    plt.figure(figsize=fig_size)
    plt.title(f"{label}")
    plt.imshow(convert_image_tensor_to_numpy_hwc_format(image_data))
