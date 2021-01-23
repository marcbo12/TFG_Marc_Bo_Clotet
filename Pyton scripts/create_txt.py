import pandas as pd
import os

'''
This code gets the path of each frame and saves it in a txt file with the yaw, pitch and roll
angle and its x and y position.
'''

rootM = '/mnt/gpid07/users/marc.bo/DATASET/Male_models/'
rootF = '/mnt/gpid07/users/marc.bo/DATASET/Female_models/'


dirsF = os.listdir(rootF)
dirsM = os.listdir(rootM)
dirTrajM = []
dirTrajF = []
for dire in dirsM:
    dirTrajM.append(dire)
for dire in dirsF:
    dirTrajF.append(dire)

for directory in dirTrajF:
    path = rootF + directory
    for dire in os.listdir(path):
        info_img = path + '/' + dire + '/' + directory + '_' + dire + '.txt'
        path_img = path + '/' + dire + '/rgb/_img'
        path_depth = path + '/' + dire + '/depth/_depth'

        try:
            f   = open(info_img,"r")
            content = f.readlines()
        except OSError:
            continue

        ang1 = []
        ang2 = []
        num = []

        for line in content:
            ang1 = []
            if line.find('Point:'):
                punto = line[line.find('Point: (') + 8:line.find(')')]
                point = punto.split(',')
                if 0.0 < float(point[0]) < 640.0:
                    if 0.0 < float(point[1]) < 480.0:
                        if line.find('Frame:'):
                            frame = line[line.find('Frame:') + 6:line.find(' Point')]
                            if float(frame) % 10 == 0:
                                ang1.append(frame)
                                num.append(float(ang1[0]))
                                ang1.append([float(point[0]), float(point[1])])
                                if line.find('Pitch'):
                                    pitch =  line[line.find('Pitch:') + 7:line.find(' Yaw')]
                                    pitch = pitch.replace(',', '.')
                                    ang1.append(float(pitch))
                                if line.find('Yaw:'):
                                    yaw = line[line.find('Yaw:') + 5:line.find(' Roll')]
                                    yaw = yaw.replace(',', '.')
                                    ang1.append(float(yaw))
                                if line.find('Roll'):
                                    roll = line[line.find('Roll:') + 6:line.find('\n')]
                                    roll = roll.replace(',', '.')
                                    ang1.append(float(roll))
                                ang2.append(ang1)

        df = pd.DataFrame(ang2, columns=['frame', 'pos', 'angle1', 'angle2', 'angle3'])
        print(df)

        paths = []

        for root, dirs, files in os.walk(path_img):
            for name in files:
                """
                if name[0:6] == '_depth':
                    paths.append(name)
                    """
                if name[0:4] == '_img':
                    paths.append(name)

        paths.sort()

        join = []

        for dire2 in ang2:
            join.append(path_img + dire2[0] + '.png')

        with open('paths/{}_{}F.txt'.format(directory, dire), "a") as t:
            for item in range(len(join)):
                if df['angle1'][item] != 0 or df['angle2'][item] != 0 or df['angle3'][item] != 0:
                    t.write(join[item] + ' ' + str(df['angle1'][item]) + ' ' + str(df['angle2'][item]) + ' ' + str(df['angle3'][item]) + ' ' + str(df['pos'][item]) + '\n')

        t.close()


for directory in dirTrajM:
    path = rootM + directory
    for dire in os.listdir(path):
        info_img = path + '/' + dire + '/' + directory + '_' + dire +'.txt'
        path_img = path + '/' + dire + '/rgb/_img'
        path_depth = path + '/' + dire + '/depth/_depth'
        try:
            f   = open(info_img,"r")
            content = f.readlines()
        except OSError:
            continue

        ang1 = []
        ang2 = []
        num = []

        for line in content:
            ang1 = []
            if line.find('Point:'):
                punto = line[line.find('Point: (') + 8:line.find(')')]
                point = punto.split(',')
                if 0.0 < float(point[0]) < 640.0:
                    if 0.0 < float(point[1]) < 480.0:
                        if line.find('Frame:'):
                            frame = line[line.find('Frame:') + 6:line.find(' Point')]
                            if float(frame) % 10 == 0:
                                ang1.append(frame)
                                num.append(float(ang1[0]))
                                ang1.append([float(point[0]), float(point[1])])
                                if line.find('Pitch'):
                                    pitch = line[line.find('Pitch:') + 7:line.find(' Yaw')]
                                    pitch = pitch.replace(',', '.')
                                    ang1.append(float(pitch))
                                if line.find('Yaw:'):
                                    yaw = line[line.find('Yaw:') + 5:line.find(' Roll')]
                                    yaw = yaw.replace(',', '.')
                                    ang1.append(float(yaw))
                                if line.find('Roll'):
                                    roll = line[line.find('Roll:') + 6:line.find('\n')]
                                    roll = roll.replace(',', '.')
                                    ang1.append(float(roll))
                                ang2.append(ang1)

        df = pd.DataFrame(ang2, columns=['frame', 'pos', 'angle1', 'angle2', 'angle3'])
        print(df)

        paths = []

        for root, dirs, files in os.walk(path_img):
            for name in files:
                """
                if name[0:6] == '_depth':
                    paths.append(name)
                    """

                if name[0:4] == '_img':
                    paths.append(name)


        paths.sort()

        join = []

        for dire2 in ang2:
            join.append(path_img + dire2[0] + '.png')

        with open('paths/{}_{}M.txt'.format(directory, dire), "a") as t:
            for item in range(len(join)):
                if df['angle1'][item] != 0 or df['angle2'][item] != 0 or df['angle3'][item] != 0:
                    t.write(join[item] + ' ' + str(df['angle1'][item]) + ' ' + str(df['angle2'][item]) + ' ' + str(df['angle3'][item]) + ' ' + str(df['pos'][item]) + '\n')

        t.close()
