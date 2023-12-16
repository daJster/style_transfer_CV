import numpy as np
import tensorflow as tf
import time
from PIL import Image
from codecarbon import EmissionsTracker

from param import filter_files_by_names, print_param, \
                    RESIZE_HEIGHT, \
                    N_ITER, \
                    CHECK_ITER, \
                    LEARNING_RATE, \
                    STYLE_WEIGHT, \
                    CONTENT_WEIGHT, \
                    MERGE_METHOD, \
                    APPLY_COLOR_PRESERVATION, \
                    CHECK_LOSS
                        
from loss import compute_loss
from model import get_model, get_optimizer
from image_processing import preprocess_image, get_result_image_size, deprocess_image, original_color_transform
from plot import plot_losses, plot_features, create_video
from merge import merge_style_features_percentage, merge_style_features_mean, merge_style_features_alternate
    

def main() :
    print_param()
    # using GPU if available
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("No GPU available. Please install GPU version of TensorFlow.")
    
    # Prepare content, style images
    name_content = "paris"
    names_style = ["starry_night"]
    content_image_path = f'../dataset/content/{name_content}.jpg'
    style_image_paths = filter_files_by_names(f'../dataset/style', names_style)


    result_height, result_width = get_result_image_size(content_image_path, RESIZE_HEIGHT)
    print("result resolution: (%d, %d)" % (result_height, result_width))
    
    
    # Preprocessing
    content_tensor = preprocess_image(content_image_path, result_height, result_width)

    # without color preservation
    style_tensors_list = [preprocess_image(style_path, result_height, result_width) for style_path in style_image_paths]

    # color preservation (color matching)
    # style_tensor1 = preprocess_image_with_color_matching(content_image_path, style_image_path_2, result_height, result_width)

    # content image rgb for color preservation
    content_image_rgb = np.array(Image.open(content_image_path))

    # generated_image = tf.Variable(tf.random.uniform(content_tensor.shape, dtype=tf.dtypes.float32)) # gaussian noise
    generated_image = tf.Variable(preprocess_image(content_image_path, result_height, result_width)) # content image

    # Build model
    model = get_model()
    optimizer = get_optimizer(LEARNING_RATE)
    print(model.summary())

    content_features = model(content_tensor)
    style_features_list = [model(style_tensor) for style_tensor in style_tensors_list]
    
    print('before merging styles')
    if len(style_features_list) == 1 :
        style_features = style_features_list[0]
    elif MERGE_METHOD == 'mean' :
        style_features = merge_style_features_mean(style_features_list)
    elif MERGE_METHOD == 'percentage' :
        style_features = merge_style_features_percentage(style_features_list, np.ones(len(style_features_list))) # choose percentages
    elif MERGE_METHOD == 'alternate' :
        style_features = merge_style_features_alternate(style_features_list)
    else :
        raise ValueError('MERGE_METHOD must be "mean", "percentage" or "alternate"')

    plot_features(style_features["block1_conv1"], "block1_conv1")
    print('styles merged')

    start_time = time.time()
    losses = {"total_loss" : [],
              "loss_content" : [],
              "loss_style" : [],
              "loss_total_variation" : [],}
    

    
    
    

    
    # Training loop
    for iter in range(N_ITER):
        with tf.GradientTape() as tape:
            total_loss, loss_content, loss_style, loss_total_variation = compute_loss(model, generated_image, content_features, style_features, result_width)

        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])

        print("iter: %4d, loss: %8.f" % (iter, total_loss))
        
        # saving losses
        if (iter+1) % CHECK_LOSS == 0 :
            losses["total_loss"].append(total_loss.numpy())
            losses["loss_content"].append(loss_content.numpy())
            losses["loss_style"].append(loss_style.numpy())
            losses["loss_total_variation"].append(loss_total_variation.numpy())
            
        if (iter + 1) % CHECK_ITER == 0:
            plot_losses(losses, LEARNING_RATE)
            generated_image_rgb = deprocess_image(generated_image, result_height, result_width)

            if APPLY_COLOR_PRESERVATION:
                # Apply the color preservation
                preserved_color_image = original_color_transform(content_image_rgb, generated_image_rgb)
                preserved_image_name = "result/image_at_iteration_%d.png" % (iter + 1)
                Image.fromarray(preserved_color_image).save(preserved_image_name)
            else:
                # Save without color preservation
                Image.fromarray(generated_image_rgb).save("result/image_at_iteration_%d.png" % (iter + 1))
    
    
    
    
    
    # Save final image after completing all iterations
    final_generated_image_rgb = deprocess_image(generated_image, result_height, result_width)
    
    if APPLY_COLOR_PRESERVATION:
        final_preserved_color_image = original_color_transform(content_image_rgb, final_generated_image_rgb)
        Image.fromarray(final_preserved_color_image).save(f"result/final_result_alpha-{CONTENT_WEIGHT}_beta-{STYLE_WEIGHT}_.png")
    else:
        Image.fromarray(final_generated_image_rgb).save(f"result/final_result_alpha-{CONTENT_WEIGHT}_beta-{STYLE_WEIGHT}_.png")

    final_time = time.time() - start_time
    print('generation of image lasted : ', final_time, 's')
    # create_video('./result/', f'./result/result_video_{name_content}.mp4')

    

if __name__ == "__main__":
    et = EmissionsTracker("Gatys")
    et.start()
    main()
    emissions = et.stop()
    print(f"Emissions : {emissions} kg of CO2.")