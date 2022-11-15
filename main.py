import cv2
import matplotlib.pyplot as plt
import math
import os, time, random
import pandas as pd
import numpy as np
import pprint

from mpl_toolkits import mplot3d

IMAGE_HEIGHT = 3024
IMAGE_WIDTH = 4032
LANE_WIDTH = 9
CLAIMED_DISTANCE = 30
FOCAL_LENGTH = 3154


####################### FUNCS FOR BOTH ANALYSIS AND IMPLEMENTATION ##########################
# given two pixel values (the center of the image and the center of the desired object bounding box)
# the angle in which the object is relative to the ray perpendicular to the sensor is found through simple geometry
def find_angle(xmin, xmax, center_of_image):
    bb_width = (xmax - xmin) / 2  # find the bb width to find the center of the object
    x_bb_center = xmin + bb_width  # find center of bounding box
    # find the angle to the object localized by object detector
    return math.atan(math.fabs(x_bb_center - center_of_image[0]) / FOCAL_LENGTH) * (180 / math.pi)


def IoU(candidate, verifier):
    # coordinates of the area of intersection.
    ix1 = np.maximum(candidate[0], verifier[0])
    iy1 = np.maximum(candidate[1], verifier[1])
    ix2 = np.minimum(candidate[2], verifier[2])
    iy2 = np.minimum(candidate[3], verifier[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = candidate[3] - candidate[1] + 1
    gt_width = candidate[2] - candidate[0] + 1

    # Prediction dimensions.
    pd_height = verifier[3] - verifier[1] + 1
    pd_width = verifier[2] - verifier[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou, (ix2, iy2, ix1, iy1)


####################################### DATA ANALYSIS #######################################
def approximate_theta_c_from_theta_v(d_claimed, theta_v):
    s_v = LANE_WIDTH / math.atan(theta_v * (math.pi / 180))
    print(math.atan(12 / 10))
    return math.atan(LANE_WIDTH / (d_claimed - s_v)) * (180 / math.pi)


def translate(theta_c_hat, theta_v, center_of_bb_x):
    # find x_c_hat
    return (math.tan(theta_c_hat * math.pi / 180) / math.tan(theta_v * math.pi / 180)) * center_of_bb_x + IMAGE_WIDTH/2


def test_translate():
    theta_c_hat = approximate_theta_c_from_theta_v(W=12, d_claimed=50, theta_v=23.71)
    x_c_hat = translate(theta_c_hat, theta_v=23.71, center_of_bb_x=3401 - IMAGE_WIDTH / 2)
    print(x_c_hat)


# gathering labels for candidate or verifier, specified by the function input argument
# create a dictionary in a dictionary, moreover, first key is the respective distance,
# second key is the respective image label, value is the data associated with said image label
def gather_label_paths(can_veri, dist_directory):
    directory = os.getcwd() + "\\" + dist_directory + "\\" + can_veri + "\\"
    distance_files = []
    for distance_file_name in os.listdir(directory):
        if distance_file_name.strip('ft').isdigit():
            # if the file name is one of the distance files then we can go in otherwise ignore it
            distance_files.append(distance_file_name)
    label_dict = {}
    # we should be in 0ft//out_labels
    # fix for the first case where there is no prediction on an image
    for distance in distance_files:
        distance_numeric = int(distance.strip('ft'))
        for label in os.listdir(directory + distance + '\\out_labels\\'):
            data = []
            with open(directory + distance + '\\out_labels\\' + label) as f:
                data.append(f.read().strip('\n'))
            f.close()
            if distance_numeric not in label_dict:
                label_dict[distance_numeric] = {label: data}
            if label not in label_dict[distance_numeric] and data != ['']:
                label_dict[distance_numeric][label] = data

    return label_dict


def get_avgs(dic):
    nested_dic = {}
    two_d_data = []
    for distance_bin in dic:
        counter = 0
        x_center_sum, y_center_sum, bb_width_sum, bb_height_sum, angle = 0, 0, 0, 0, 0
        for file in dic[distance_bin]:
            # sum all predictions
            if dic[distance_bin][file] != ['']:
                x_center_sum += float(dic[distance_bin][file][0].split(' ')[1])
                y_center_sum += float(dic[distance_bin][file][0].split(' ')[2])
                bb_width_sum += float(dic[distance_bin][file][0].split(' ')[3])
                bb_height_sum += float(dic[distance_bin][file][0].split(' ')[4])
                entry = dic[distance_bin][file][0].split(' ')[7].strip()
                print(file)
                angle += float(entry)
                counter += 1
        # average bounding boxes
        x_center_avg = x_center_sum / counter
        y_center_avg = y_center_sum / counter
        bb_width_avg = bb_width_sum / counter
        bb_height_avg = bb_height_sum / counter
        angle = angle / counter
        nested_dic.update({distance_bin: [x_center_avg, y_center_avg, bb_width_avg,
                                          bb_height_avg, angle]})  # nested dictionary disregarding the file name
        # two_d_data.append([can_ver_str, distance_bin, x_center_avg, y_center_avg, bb_width_avg, bb_height_avg])
    return nested_dic


def df_for_avgs():
    # go through all files, for each distance bin create a file create a pandas dataframe [candidate/verifier,
    # 50/100, distance_bin, average_bb_center, avg_bb_width, avg_bb_height]
    header = ['candidate or verifier', 'distance_bin', 'x_center_avg', 'y_center_avg', 'bb_width_avg', 'bb_height_avg']
    can_dic = gather_label_paths("candidate", "100_feet_imgs")
    ver_dic = gather_label_paths("verifier", "100_feet_imgs")

    # can_dic = gather_label_paths("candidate", "50_feet_imgs")
    # ver_dic = gather_label_paths("verifier", "50_feet_imgs")
    # data_dict = {'candidate': can_dic, 'verifier': ver_dic}
    nested_dic_can = get_avgs(can_dic)  # get average predictions in the form of a dictionary whose key is a bin
    nested_dic_ver = get_avgs(ver_dic)  # get average predictions
    data_dic = {'candidate': nested_dic_can, 'verifier': nested_dic_ver}
    # df = pd.DataFrame(two_d_data_can + two_d_data_ver)
    # df.columns = header
    # df = df.set_index(['candidate or verifier', 'distance_bin'])
    # converting data_dict so a multi indexed dataframe can be utilized
    reform = {(outerKey, innerKey): values for outerKey, innerDict in data_dic.items() for innerKey, values in
              innerDict.items()}
    df = pd.DataFrame(reform)
    return df


def rotate_bounding_box(candidate_prediction):
    # careful with the indices here
    bb_center, width, height = (candidate_prediction[1], candidate_prediction[2]), candidate_prediction[3], \
                               candidate_prediction[4]  # convert_from_yolo_to_norm(candidate_prediction)
    # # find the quadrant where the bounding box lives, then flip into the respective quadrant
    distance_from_x_axis = abs(bb_center[1] - IMAGE_HEIGHT / 2)
    distance_from_y_axis = abs(bb_center[0] - IMAGE_WIDTH / 2)
    # if top left quad
    if bb_center[0] < IMAGE_WIDTH / 2 and bb_center[0] < IMAGE_HEIGHT / 2:
        new_bb = (bb_center[0] + 2 * distance_from_y_axis, bb_center[1] + 2 * distance_from_x_axis)
    # if top right quad
    elif IMAGE_WIDTH / 2 < bb_center[0] < IMAGE_HEIGHT / 2:
        new_bb = (bb_center[0] - 2 * distance_from_y_axis, bb_center[1] + 2 * distance_from_x_axis)
    # if bot left quad
    elif IMAGE_WIDTH / 2 > bb_center[0] > IMAGE_HEIGHT / 2:
        new_bb = (bb_center[0] + 2 * distance_from_y_axis, bb_center[1] - 2 * distance_from_x_axis)
    # if bot right quad
    else:
        new_bb = (bb_center[0] - 2 * distance_from_y_axis, bb_center[1] - 2 * distance_from_x_axis)
    top_left_corner = (int(new_bb[0] - width / 2),
                       int(new_bb[1] - height / 2))
    bot_right_corner = (int(new_bb[0] + width / 2),
                        int(new_bb[1] + height / 2))
    return top_left_corner, bot_right_corner, new_bb


def find_complementary_df(df, cand_or_dist, distance_bin, claimed_distance):
    # if cand falls in 15 ft bin veri falls in 35 ft
    # cand_or_dist is str that is either 'candidate' or 'verifier'
    # distance_claimed is an int that refers to the claimed distance
    # distance_bin is an int referring to the distance from cand to target vehicle
    complementary_distance_bin = claimed_distance - distance_bin
    print()
    if cand_or_dist == 'candidate':
        # print('Verifier', complementary_distance_bin, 'complements candidate', distance_bin, 'away from target vehicle\n',df[('verifier', complementary_distance_bin)])
        return df[('verifier', complementary_distance_bin)]
    else:
        # print('Candidate', complementary_distance_bin, 'complements verifier', distance_bin, 'away from target vehicle\n',df[('candidate', complementary_distance_bin)])
        return df[('candidate', complementary_distance_bin)]


def find_complementary_image(cand_or_dist, distance_bin, claimed_distance):
    pass


def calculate_center_point_error(data_dataframe, claimed_distance):
    # iterate through all the CANDIDATE data points
    # ('candidate', verifier)
    candidate_second_key = [15, 20, 25, 30, 35, 40, 45]  # target is this far away from the candidate
    euclidean_distance = []
    iou_list = []
    i = 0
    for index in candidate_second_key:
        cand_df = data_dataframe[('candidate', index)]
        complementary_dataframe = find_complementary_df(df, 'candidate', index, claimed_distance)
        # un-normalizing
        verifier_prediction = [0, IMAGE_WIDTH * complementary_dataframe[0],
                               IMAGE_HEIGHT * complementary_dataframe[1],
                               IMAGE_WIDTH * complementary_dataframe[2],
                               IMAGE_HEIGHT * complementary_dataframe[3], complementary_dataframe[4]]
        candidate_prediction = [0, IMAGE_WIDTH * cand_df[0], IMAGE_HEIGHT * cand_df[1], IMAGE_WIDTH * cand_df[2],
                                IMAGE_HEIGHT * cand_df[3], cand_df[4]]
        # original rotated approach
        top_left_corner, bottom_right_corner, new_center = rotate_bounding_box(
            candidate_prediction)  # rotate the candidate bounding box
        pixel_dis = pixel_distance((new_center[0], new_center[1]), (verifier_prediction[1], verifier_prediction[2]))
        euclidean_distance.append(pixel_dis)
        candidate_per_rotated_ver_per = [top_left_corner[0], top_left_corner[1], bottom_right_corner[0],
                                         bottom_right_corner[1]]
        cand_gt = [candidate_prediction[1] - candidate_prediction[3] / 2,
                  candidate_prediction[2] - candidate_prediction[4] / 2,
                  candidate_prediction[1] + candidate_prediction[3] / 2,
                  candidate_prediction[2] + candidate_prediction[4] / 2]
        ver_gt = [verifier_prediction[1] - verifier_prediction[3] / 2,
                  verifier_prediction[2] - verifier_prediction[4] / 2,
                  verifier_prediction[1] + verifier_prediction[3] / 2,
                  verifier_prediction[2] + verifier_prediction[4] / 2]
        # another approach by trying to scale
        center_of_bb_x = verifier_prediction[1] # center of the bounding box from candidate perspective
        theta_c_hat = approximate_theta_c_from_theta_v(d_claimed=50, theta_v=verifier_prediction[5])
        x_c_hat = translate(theta_c_hat=theta_c_hat,theta_v=verifier_prediction[5], center_of_bb_x=center_of_bb_x) # gets center of bb in image domain from verifier perspective
        cand_pred = [(int(candidate_prediction[1] - candidate_prediction[3]/2), int(candidate_prediction[2] - candidate_prediction[4]/2)),
                     (int(candidate_prediction[1] + candidate_prediction[3]/2), int(candidate_prediction[2] + candidate_prediction[4]/2))]
        new_prediction = [
            (int(x_c_hat - verifier_prediction[3] / 2), int(verifier_prediction[2] - verifier_prediction[4] / 2)),
            (int(x_c_hat + verifier_prediction[3] / 2), int(verifier_prediction[2] + verifier_prediction[4] / 2))]
        # iou, common_area = IoU(ver_gt, candidate_per_rotated_ver_per)
        iou, common_area = IoU(cand_gt, [int(x_c_hat - verifier_prediction[3] / 2),
                                        int(verifier_prediction[2] - verifier_prediction[4] / 2),
                                        int(x_c_hat + verifier_prediction[3] / 2),
                                        int(verifier_prediction[2] + verifier_prediction[4] / 2)])
        if index == 45:
            # 'C:\\Users\\Lewis\\PycharmProjects\\torch_yolov5\\50_feet_imgs\\verifier\\35ft\\out_images\\IMG_9859.JPG'
            draw_bounding_boxes_on_verifier('50_feet_imgs/candidate/45ft/out_images/IMG_8641.JPG',
                                            new_prediction[0], new_prediction[1],
                                            (0, 0, 255))
            # draw_bounding_boxes_on_verifier('test\\new_method_45_ft_from_cand.jpg', cand_pred[0], cand_pred[1],
            #                                 (255, 0, 0))
            # draw_bounding_boxes_on_verifier('test\\new_method_45_ft_from_cand.jpg',
            #                                 (int(common_area[0]), int(common_area[1])),
            #                                 (int(common_area[2]), int(common_area[3])), (0, 255, 0))

        iou_list.append(iou)
    plot_pixel_dist(candidate_second_key, euclidean_distance)


# bb_center is a tuple consisting of x and y frame pixels that are the center of bb in frame
# prediction is yolov5 format so the bb_center will be normalized to the overall size of the frame
def find_angle(bb_center):
    math.fabs(bb_center[0] - IMAGE_WIDTH / 2)
    pass


def pixel_distance(coordinate_1, coordinate_2):
    # distance between pixels
    return math.sqrt((coordinate_1[0] - coordinate_2[0]) ** 2 + (coordinate_1[1] - coordinate_2[1]) ** 2)


def average_angle(veh_dict):
    avg_angle = []
    for key in veh_dict:
        sum = 0
        iteration = 0
        for label in veh_dict[key]:
            sum += float(veh_dict[key][label][0].split(' ')[7])  # 7 is the angle prediction
            iteration += 1
        avg = sum / iteration
        dist = int(key.strip('ft\n'))
        avg_angle.append((dist, avg))
    return avg_angle


def average_hypotenuse(avg_angle):
    total_estimation = []
    for pair in avg_angle:
        total_estimation.append((pair[0], LANE_WIDTH / math.tan(pair[1] * math.pi / 180), pair[1]))
    return total_estimation


def draw_bounding_boxes(image, prediction):
    # prediction is in the form of a pandas dataframe, moreover, iterate through each row and populate bounding boxes
    # bounding box information must
    # indexing through the rows
    im = cv2.imread(image)
    for i in range(prediction.shape[0]):
        xmin, ymin, xmax, ymax = prediction.iloc[i]['xmin'], prediction.iloc[i]['ymin'], prediction.iloc[i]['xmax'], \
                                 prediction.iloc[i]['ymax']
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        color = (0, 0, 255)
        cv2.rectangle(im, pt1, pt2, color, 2)
    cv2.imwrite('C:\\Users\\Lewis\\PycharmProjects\\torch_yolov5\\test\\test.jpg', im)


def convert_bounding_box_info(bounding_box_string):
    # converting from YOLOv5 format to a decimal representation
    # class, x_center, y_center, w, h, confidence, distance note: x,y,w,h are all normalized
    return bounding_box_string.split(' ')[:5]  # should cut off info regarding confidence and distance


def gather_ground_truth(dist_directory):
    y = [[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
         [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]]
    ground_truth = []
    if dist_directory == '50_feet_images':
        i = 0
    else:
        i = 1
    for depth in y[i]:
        hypotenuse = math.sqrt(LANE_WIDTH ** 2 + depth ** 2)
        angle = math.atan(LANE_WIDTH / depth) * 180 / math.pi
        # append a tuple in the form of depth, hypotenuse, and angle
        ground_truth.append((depth, hypotenuse, angle))
    return ground_truth


def create_2d_map(theta_v_s):
    x = [0]
    y = [0]
    for theta_v in theta_v_s:
        x.append(LANE_WIDTH)
        y.append((LANE_WIDTH / math.tan((theta_v * math.pi) / 180)))
    plt.xlim(-20, 20)
    plt.ylim(-20, 100)
    plt.grid()
    plt.plot(x, y, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    plt.savefig("mygraph.png")


def error(ground_truth, estimates):
    x = []
    hypotenuse_error = []
    angle_error = []
    # ground_truth = ground_truth[2:]
    for i in range(len(estimates)):
        x.append(estimates[i][0] - 5)
        hypotenuse_error.append(math.fabs((estimates[i][1] - ground_truth[i][1]) / (ground_truth[i][1])) * 100)
        angle_error.append(math.fabs((estimates[i][2] - ground_truth[i][2]) / (ground_truth[i][2])) * 100)
    print(x)
    print(hypotenuse_error)
    print(angle_error)
    return x, hypotenuse_error, angle_error


###############################################################################################################
######################################### PLOTTING ##############################################################

def plot_error(x, hypotenuse_error, angle_error):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    st = fig.suptitle("Candidate Error", fontsize="x-large")

    ax[0].set_title('Hypotenuse Error For Candidate Given 50 Foot Range')
    ax[0].set_xlabel('True Target Vehicle Depth in Feet (Not True Hypotenuse Distance)')
    ax[0].set_ylabel('Error in Percentage')
    plt.grid()
    ax[0].plot(x, hypotenuse_error, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="red")
    ax[1].set_title('Angle Error For Candidate Given 50 Foot Range')
    ax[1].set_xlabel('True Target Vehicle Depth in Feet (Not True Hypotenuse Distance)')
    ax[1].set_ylabel('Error in Percentage')
    plt.grid()
    ax[1].plot(x, angle_error, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="red")
    plt.show()


def plot_pixel_dist(x, pixel_dist):
    # make data
    fig, ax = plt.subplots()
    ax.plot(x, pixel_dist, linewidth=2.0)
    plt.title('Pixel Distance of the Center of the Rotated Candidate Bounding Box')
    plt.ylabel('Pixel Distance from Rotated Candidate Bounding Box Center to Verifier Bounding Box Center (Pixels)')
    plt.xlabel('Target Vehicle Distance from Candidate (Feet)')
    plt.show()


###############################################################################################################
######################################### MIRROR ##############################################################

def convert_from_yolo_to_norm(cand_list):
    cand_list = cand_list.strip('\\n').split(' ')
    bb_center = (int(IMAGE_WIDTH * float(cand_list[1])), int(IMAGE_HEIGHT * float(cand_list[2])))
    # top_left_corner = (int(IMAGE_WIDTH * float(cand_list[1]) - IMAGE_WIDTH * float(cand_list[3]) / 2),
    #                    int(IMAGE_HEIGHT * float(cand_list[2]) - IMAGE_HEIGHT * float(cand_list[4]) / 2))
    # bot_right_corner = (int(IMAGE_WIDTH * float(cand_list[1]) + IMAGE_WIDTH * float(cand_list[3]) / 2),
    #                     int(IMAGE_HEIGHT * float(cand_list[2]) + IMAGE_HEIGHT * float(cand_list[4]) / 2))
    return bb_center, int(IMAGE_WIDTH * float(cand_list[3])), int(IMAGE_HEIGHT * float(cand_list[4]))


def draw_bounding_boxes_on_verifier(verifier_image, top_left_corner, bot_right_corner, color):
    # draw a new bounding box
    im = cv2.imread(verifier_image)
    # color = (0, 0, 255)
    cv2.rectangle(im, top_left_corner, bot_right_corner, color, 5)
    # cv2.putText(im, 'Rotated Bounding Box', (top_left_corner[0], top_left_corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,(36, 255, 12), 2)
    cv2.imwrite('test\\new_method_45_ft_from_cand.jpg', im)


df = df_for_avgs()
print(df)
# complementary_df = find_complementary_image(df, 'candidate', 15, 50)
# calculate_center_point_error(df, 50)


# draw_bounding_boxes_on_verifier(verifier_image, top_left_corner, bot_right_corner)
# rotate_bounding_box(
#     'C:\\Users\\Lewis\\PycharmProjects\\torch_yolov5\\50_feet_imgs\\verifier\\30ft\\out_images\\IMG_9870.JPG',
#     candidate_prediction="0 0.900422 0.520337 0.193204 0.241733 0.893444 7.2 27.12")


###############################################################################################
####################################### IMPLEMENTATION ########################################
# path_to_weights = 'weights_for_mutual_sensing.pt'
# # os.system("pip install -r requirements.txt") # RUN ONCE PRIOR TO LOADING MODEL
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights, force_reload=False)  # default
# model.conf = 0.60
# image_path = "C:\\Users\\Lewis\\PycharmProjects\\torch_yolov5\\50_feet_imgs\\candidate\\50ft\\IMG_8721.JPG"
# pp = pprint.PrettyPrinter(width=41, compact=True)

# create a 2d plot showing the bounding box trace
# moreover, x-y plane of the plot is the image, said plot will
# show the movement of the center along the x-y plane when the
# car moves further and closer to the car
def two_dimensional_bounding_box_trace(candidate_predictions, verifier_predictions, size_bin):
    # firstly, let us populate the x and y points
    # get center bounding box average for two-dimensional bounding box trace
    if size_bin == 100:
        distances_cand = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        distances_ver = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    else:
        distances_cand = [15, 20, 25, 30, 35, 40, 45, 50]
        distances_ver = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cand_x = []
    ver_x = []
    cand_y = []
    ver_y = []
    for distance in distances_cand:
       cand_x.append(candidate_predictions[distance][0] * IMAGE_WIDTH)
       cand_y.append(-1*candidate_predictions[distance][1] * IMAGE_HEIGHT)

    for distance in distances_ver:
        ver_x.append(verifier_predictions[distance][0] * IMAGE_WIDTH)
        ver_y.append(-1*verifier_predictions[distance][1] * IMAGE_HEIGHT)

    plt.plot(ver_x, ver_y,'o', label='Verifier')
    plt.plot(cand_x, cand_y, 'o', label='Candidate')
    plt.xlabel('Horizontal Axis of Image')
    plt.ylabel('Vertical Axis of Image')
    plt.title('2-Dimensional Bounding Box Center Trace')
    plt.legend(loc="upper left")
    plt.ylim(-IMAGE_HEIGHT, 0)
    plt.xlim(0, IMAGE_WIDTH)

    plt.savefig('2-Dimensional Bounding Box Trace')

two_dimensional_bounding_box_trace(df['candidate'], df['verifier'], size_bin=100)

# create a 3d plot showing the bounding box trace
# moreover, x-y plane of the plot is the image, and the axis
# perpendicular to the x-y plane is the actaul real life
# distance away from the car
def three_dimensional_bounding_box_trace(candidate_predictions, verifier_predictions, size_bin):
    # firstly, let us populate the x and y points
    # get center bounding box average for two-dimensional bounding box trace
    if size_bin == 100:
        distances_cand = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        distances_ver = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    else:
        distances_cand = [15, 20, 25, 30, 35, 40, 45, 50]
        distances_ver = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cand_x = []
    ver_x = []
    cand_y = []
    ver_y = []
    for distance in distances_cand:
        cand_x.append(candidate_predictions[distance][0] * IMAGE_WIDTH)
        cand_y.append(-1 * candidate_predictions[distance][1] * IMAGE_HEIGHT)

    for distance in distances_ver:
        ver_x.append(verifier_predictions[distance][0] * IMAGE_WIDTH)
        ver_y.append(-1 * verifier_predictions[distance][1] * IMAGE_HEIGHT)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(ver_x, ver_y, distances_ver,  'o', label='Verifier')
    ax.plot3D(cand_x, cand_y, distances_cand,  'o', label='Candidate')

    plt.xlabel('Horizontal Axis of Image')
    plt.ylabel('Vertical Axis of Image')
    plt.title('3-Dimensional Bounding Box Center Trace')
    plt.legend(loc="upper left")
    plt.ylim(-IMAGE_HEIGHT, 0)
    plt.xlim(0, IMAGE_WIDTH)

    plt.show()

three_dimensional_bounding_box_trace(df['candidate'], df['verifier'], size_bin=100)

def find_angle(xmin, xmax, center_of_image):
    bb_width = (xmax - xmin) / 2
    x_bb_center = xmin + bb_width
    return math.atan(math.fabs(x_bb_center - center_of_image[0]) / FOCAL_LENGTH) * (180 / math.pi)


def predict_and_2d_localize(image_path):
    results = model(image_path)
    pandas_pred = results.pandas().xyxy[0]
    pp.pprint(pandas_pred)
    # im = Image.open(image_path)
    # im.show()
    draw_bounding_boxes(pandas_pred, image_path)
    center_of_image = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)
    angles = []
    for i in range(pandas_pred.shape[0]):
        xmin, ymin, xmax, ymax = pandas_pred.iloc[i]['xmin'], pandas_pred.iloc[i]['ymin'], pandas_pred.iloc[i]['xmax'], \
                                 pandas_pred.iloc[i]['ymax']
        approximated_angle = find_angle(xmin, xmax, center_of_image)
        angles.append(approximated_angle)
    create_2d_map(angles)

    # print(results)

###############################################################################################
# cand_dict = gather_label_paths(can_veri="candidate", dist_directory="50_feet_imgs")
# # create_2d_map(theta_v=30.93)
# ground_truth = gather_ground_truth(dist_directory='50_feet_images')
# print("Ground Truth:")
# print(ground_truth)
# collective_estimations = average_hypotenuse(average_angle(cand_dict))
# print("Estimates:")
# print(collective_estimations)
# x, hyp_err, angle_error = error(ground_truth, collective_estimations)
# # plot_error(x, hyp_err, angle_error)
# predict_and_2d_localize(image_path)
