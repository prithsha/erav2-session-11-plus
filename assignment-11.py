# %%
# Download the Repository 
# !git clone "https://github.com/prithsha/erav2-session-11-plus.git"


# %%
# Change to working directory as needed
# %cd /content/<>

# %%
# Install the requirements
# !pip install -r ./requirements.txt

# %%
import main
from utility import cifar10Utility
from utility import imageAugmentationUtility
from utility import imageVisualizationUtility

# %%

train_transforms, test_transforms = imageAugmentationUtility.get_cifar10_train_and_test_transforms(cifar10Utility.get_mean(),
                                                                                                   cifar10Utility.get_std())



# %%
BATCH_SIZE = 512
DATA_FOLDER = "./data"

train_dataset, test_dataset = cifar10Utility.get_datasets(train_transforms_collection=train_transforms,
                                                                   test_transforms_collection=test_transforms,
                                                                    data_folder=DATA_FOLDER)
train_loader, test_loader = cifar10Utility.get_dataloaders(train_dataset=train_dataset,
                                                                    test_dataset=test_dataset,
                                                                    batch_size=BATCH_SIZE)

# %%
imageVisualizationUtility.randomly_show_images_from_tensor_array(train_dataset, cifar10Utility.get_image_classes(), fig_size=(16,4))

# %%
model = main.get_model_instance(model_type=main.ModelType.RESNET18)
optimizer = main.get_adam_optimizer(model)
scheduler = main.get_stepLR_scheduler(optimizer)
criterion = main.get_cross_entropy_loss_criteria()

# %%
EPOCHS = 2
model_executor = main.NetworkModelEvaluator(train_loader, test_loader)
model_executor.execute(epochs=EPOCHS, model=model, criterion=criterion,
                       optimizer=optimizer, scheduler=scheduler)

# %%
print(f"----****----Wrongly predicted test images: {len(model_executor.wrongly_predicted_test_images)}")
imageVisualizationUtility.show_images_from_tensor_array(list(model_executor.wrongly_predicted_test_images), cifar10Utility.get_image_classes(), shape=(2,10))

print(f"----****----Wrongly predicted train images: {len(model_executor.wrongly_predicted_trained_images)}")
imageVisualizationUtility.show_images_from_tensor_array(list(model_executor.wrongly_predicted_trained_images), cifar10Utility.get_image_classes(), shape=(2,10))

# %%
module = model.module
target_layer = module.layer3
print(target_layer)

images = list(model_executor.wrongly_predicted_test_images)

# %%
from pytorch_grad_cam import GradCAM
selected_image, label = images[13]
print(selected_image.size())
target_layers = module.layer3[-1]
print(target_layers)
cam = GradCAM(model=model, target_layers=[target_layers])

# %%
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM

def preprocess_image(image_data):
    transform = transforms.ToPILImage()
    image_data = transform(image_data)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_data).unsqueeze(0)


selected_pil_image = preprocess_image(selected_image)


# %%
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

targets  = ClassifierOutputTarget(9)
grayscale_cam = cam(input_tensor=selected_pil_image, targets=[targets])

print(grayscale_cam.shape)
grayscale_cam = grayscale_cam[0, :]
print(grayscale_cam.shape)



# %%
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

print(grayscale_cam.shape)


final_image = selected_pil_image.numpy() / 255
final_image = final_image.squeeze(0).transpose((1, 2, 0))

print(final_image.shape)
visualization = show_cam_on_image(final_image , grayscale_cam, use_rgb=True)

# %%
visualization.shape

imageVisualizationUtility.show_image(selected_image.cpu(), label.cpu(), fig_size=(2,2))

imageVisualizationUtility.show_processed_images([(visualization,0)], [label], rows=1, cols=1, fig_size=(2,2))



