import numpy as np
import cv2
import multiprocessing
import matplotlib.pyplot as plt

def redistribute(hist, clip_limit):

    top = clip_limit
    bottom = 0
    S = 0
    while (top - bottom > 1):
        middle = (top + bottom) / 2
        S = 0
        for i in range(len(hist)):
            if hist[i] > middle:
                S += (hist[i]-middle)
        if S > ((clip_limit - middle)*len(hist)):
            top = middle
        else:
            bottom = middle

    return (S/len(hist))+bottom


def clahe_cdf(Iay):
    [H, W] = Iay.shape
    NM = W*H
    EqIay = np.zeros((H, W))
    histoy = np.zeros(256)

    # Step 1: Calculate the image histogram
    for j in range(H):
        for i in range(W):
            valy = Iay[j][i]
            if (valy > 0 and valy < 256):
                histoy[valy] = histoy[valy]+1

    clip_limit = 3 * np.mean(histoy)

    P = redistribute(histoy, clip_limit)
    L = clip_limit - P

    total_weight = 0
    for i in range(len(histoy)):
        if histoy[i] > L:
            total_weight += (histoy[i]-L)
            histoy[i] = L

    inc = total_weight/len(histoy)

    for i in range(len(histoy)):
        histoy[i] += inc

    pdfy = np.zeros(histoy.shape)
    cdfy = np.zeros(histoy.shape)

    for i in range(len(histoy)):
        pdfy[i] = histoy[i]/NM
        cdfy[i] = pdfy[i]
        if (i > 1):
            cdfy[i] = cdfy[i]+cdfy[i-1]

    return cdfy


def calc_cdf(region):
    return clahe_cdf(region)


def main_CLAHE(input_img):

    rgbImage = input_img
    [y, cr, cb] = cv2.split(cv2.cvtColor(rgbImage, cv2.COLOR_RGB2YCrCb))
    clahe_border = 8
    new_y = cv2.copyMakeBorder(y, clahe_border, clahe_border, clahe_border, clahe_border, cv2.BORDER_REFLECT)
    clahe = np.float32(new_y)    
    coordinates_lst = []
    for i in range(clahe_border, clahe.shape[0]-clahe_border):
        for j in range(clahe_border, clahe.shape[1]-clahe_border):
                coordinates_lst.append([i,j])
            
    regions = [new_y[i[0]:i[0]+clahe_border, i[1]:i[1]+clahe_border] for i in coordinates_lst]
    
    with multiprocessing.Pool(5) as p:
        cdfs = p.map(calc_cdf, regions)

    for index, i in enumerate(coordinates_lst):
        clahe[i[0], i[1]] = cdfs[index][new_y[i[0], i[1]]]*255
        # Rayleigh Distribution
        # clahe[i[0], i[1]] = np.power(2*((100)**2)*np.log(1/(1-np.float32(cdfs[index][new_y[i[0], i[1]]]))), 0.5)
        # Expontential Distribution 
        # clahe[i[0], i[1]] = - (1/0.25)*np.log((1-cdfs[index][new_y[i[0], i[1]]]))

    clahe = clahe[clahe_border:clahe.shape[0]-clahe_border, clahe_border:clahe.shape[1]-clahe_border]
    clahe = cv2.cvtColor(
        cv2.merge((np.uint8(clahe), cr, cb)), cv2.COLOR_YCrCb2RGB)

    return clahe

if __name__ == '__main__':
    input_path = input("Input Path: ")
    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(img)

    img = main_CLAHE(img)
    plt.subplot(122)
    plt.title("CLAHE")
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig("clahe_output")
    plt.show()

