from tkinter.tix import InputOnly
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def main(input_path, output_path):

    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.title("Original Image")
    plt.imshow(img)

    [rows, columns, channels] = img.shape

    img = cv2.normalize(img, None, alpha=0.00001, beta=1,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    rgb_to_lms = np.array((np.array((0.3811, 0.5783, 0.0402)), np.array(
        (0.1967, 0.7244, 0.0782)), np.array((0.0241, 0.1288, 0.8444))))
    loglms_to_lab = np.matmul(np.array((np.array(((1/np.sqrt(3)), 0, 0)), np.array((0, (1/np.sqrt(6)), 0)),
                                        np.array((0, 0, (1/np.sqrt(2)))))), np.array((np.array((1, 1, 1)), np.array((1, 1, -2)), np.array((1, -1, 0)))))
    lab_to_lms = np.matmul(np.array((np.array((1, 1, 1)), np.array((1, 1, -1)), np.array((1, -2, 0)))),
                           np.array((np.array(((np.sqrt(3)/3), 0, 0)), np.array((0, (np.sqrt(6)/6), 0)),
                                     np.array((0, 0, (np.sqrt(2)/2))))))
    lms_to_rgb = np.array((np.array((4.4679, -3.5873, 0.1193)), np.array(
        (-1.2186, 2.3809, -0.1624)), np.array((0.0497, -0.2439, 1.2045))))

    new_img = np.zeros(img.shape)
    points = []
    rads = []
    for i in range(rows):
        for j in range(columns):
            pixel = img[i][j]
            pixel = np.dot(rgb_to_lms, pixel)
            pixel = np.log(pixel)
            pixel = np.dot(loglms_to_lab, pixel)
            points.append([pixel[1], pixel[2]])
            new_img[i][j] = [pixel[0], pixel[1], pixel[2]]
            rads.append(180+np.rad2deg(np.arctan2(pixel[2],pixel[1])))

    n_clusters = 1
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(points)
    centroids  = kmeans.cluster_centers_  
        
    plt.subplot(222)
    plt.title("Alpha-Beta of LAB")
    plt.scatter([i[0] for i in points], [i[1] for i in points], s=1.0, alpha=0.1, c=clusters)
    for i in range(n_clusters):
        plt.scatter(centroids[i][0], centroids[i][1], s=50, c='red', marker='x')


    points = []
    count = 0
    for i in range(rows):
        for j in range(columns):

            pixel = new_img[i][j]

            t = 360+np.rad2deg(np.arctan2(pixel[2],pixel[1]))
            interps = [0.0, 0.5, 0.8, 0.8, 0.5]

            interp = np.interp(t, [0, 90, 180, 270, 360], interps)
            
            index = clusters[count]
            pixel[1] -= centroids[index][0]*interp
            pixel[2] -= centroids[index][1]*interp

            points.append([pixel[1], pixel[2]])

            pixel = np.dot(lab_to_lms, pixel)
            pixel = np.exp(pixel)
            pixel = np.dot(lms_to_rgb, pixel)


            new_img[i][j] = [pixel[0], pixel[1], pixel[2]]

            count += 1

    plt.subplot(224)
    plt.title("Alpha-Beta of corrected LAB")
    centroids = np.array([points]).mean(axis=1)[0]
    plt.scatter([i[0] for i in points], [i[1] for i in points], s=0.001, c='black')
    plt.scatter(centroids[0], centroids[1], s=50, c='red', marker='x')

    img = cv2.normalize(new_img, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    plt.subplot(223)
    plt.title("Corrected Image")
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    return img


if __name__ == "__main__":
    input_path = input("Input Path: ")
    output_path = input("Output Path: ")
    img = main(input_path, output_path)
    plt.title("LAB Color Correction")
    plt.imshow(img)