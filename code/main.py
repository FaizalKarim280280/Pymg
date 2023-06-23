import pymg as pymg
import matplotlib.pyplot as plt
import cv2

def main():
    
    PATH = '../coldplay.jpg'
    img = plt.imread(PATH)
    # img = cv2.resize(img, (100, 100))
    img = pymg.load_img(PATH, between=(0, 1), retain_png= False)
    
    print(help(pymg.resize_image))
    print(help(pymg.normalize_image))
    print(help(pymg.load_img))
    
    # img = pymg.convert2gray(img)
        
    # img = pymg.discretize_mask(img, threshold=0.5)
    
    img = pymg.resize_image(img, size = (100, 100))
    
    print(img)
    
    # img = cv2.resize(img, (100, 100))
    
        
    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    main()