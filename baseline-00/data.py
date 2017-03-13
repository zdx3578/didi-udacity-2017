from utility.common import *
import pykitti


def point3d_to_top_image(lidar):
    image=None

    return image



def point3d_to_front_image(lidar):
    image=None

    return image



def project_proposal(proposal):

    return top_proposal, front_propsal, rbg_proposal




# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    basedir = '/root/share/project/didi/data/kitti/dummy'
    date  = '2011_09_26'
    drive = '0005'

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive, range(0, 50, 5))

    # Load some data
    dataset.load_calib()        # Calibration data are accessible as named tuples
    dataset.load_timestamps()   # Timestamps are parsed into datetime objects
    dataset.load_oxts()         # OXTS packets are loaded as named tuples
    dataset.load_gray()         # Left/right images are accessible as named tuples
    dataset.load_rgb()          # Left/right images are accessible as named tuples
    dataset.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]

    # Display some of the data
    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frame_range))

    print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
    print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
    print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
    print('\nSecond IMU pose:\n' + str(dataset.oxts[1].T_w_imu))

    f, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0, 0].imshow(dataset.gray[0].left, cmap='gray')
    ax[0, 0].set_title('Left Gray Image (cam0)')

    ax[0, 1].imshow(dataset.gray[0].right, cmap='gray')
    ax[0, 1].set_title('Right Gray Image (cam1)')

    ax[1, 0].imshow(dataset.rgb[0].left)
    ax[1, 0].set_title('Left RGB Image (cam2)')

    ax[1, 1].imshow(dataset.rgb[0].right)
    ax[1, 1].set_title('Right RGB Image (cam3)')

    f2 = plt.figure()
    ax2 = f2.add_subplot(111, projection='3d')
    # Plot every 100th point so things don't get too bogged down
    velo_range = range(0, dataset.velo[2].shape[0], 100)
    ax2.scatter(dataset.velo[2][velo_range, 0],
                dataset.velo[2][velo_range, 1],
                dataset.velo[2][velo_range, 2],
                c=dataset.velo[2][velo_range, 3],
                cmap='gray')
    ax2.set_title('Third Velodyne scan (subsampled)')

    plt.show()