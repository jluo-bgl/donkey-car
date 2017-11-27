import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData, AngleTypeWithZeroRecordAllocator, AngleSegmentRecordAllocator, AngleSegment
from data_load import DriveDataSet, DataGenerator, \
    drive_record_filter_exclude_duplicated_small_angles, drive_record_filter_include_all, drive_record_filter_exclude_zeros
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator, \
    flip_random_generator
from visualization import Video, Plot
from donkeycar.parts.datastore import TubHandler, TubGroup


class TestPlot(unittest.TestCase):
    @staticmethod
    def create_real_dataset(filter_method):
        tubgroup = TubGroup("../data/aws")
        return DriveDataSet.from_tubgroup(
            tubgroup.df,
            filter_method=filter_method,
            fake_image=False
        )

    def test_gen_video(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_include_all)
        Plot.video_from_datasets("angle/aws.mp4", dataset)

    def test_angle_distribution(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_include_all)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("angle/angle_distribution_original.jpg")

    def test_angle_groud_truth_prediction(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_include_all)
        plt = Plot.angle_prediction(dataset.angles(), [0])
        plt.savefig("angle/angle_ground_truth_prediction.jpg")

    def test_angle_distribution_large_speed(self):
        filter_method=drive_record_filter_include_all
        tubgroup = TubGroup("../data/log_w_6,../data/log_w_7")
        dataset = DriveDataSet.from_tubgroup(
            tubgroup.df,
            filter_method=filter_method,
            fake_image=True
        )

        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("angle/angle_distribution_original_faster_max_speed.jpg")

    def test_angle_distribution_after_filterout_small_angles(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_exclude_duplicated_small_angles)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("angle/angle_distribution_exclude_small_angles.jpg")

    def test_angle_distribution_after_filterout_zeros(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_exclude_zeros)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("angle/angle_distribution_exclude_zero_angles.jpg")

    def test_angle_distribution_generator_45_10_45_pipe_line(self):
        data_set = self.create_real_dataset(filter_method=drive_record_filter_exclude_duplicated_small_angles)
        allocator = AngleTypeWithZeroRecordAllocator(data_set, 20, 20, 15, 15, 15, 0.25)
        generator = pipe_line_generators(
            shift_image_generator(angle_offset_pre_pixel=0.002),
            flip_generator,
            brightness_image_generator(0.25)
        )
        self._angle_distribution(
            "angle_distribution_generator_exclude_duplicated_small_angles_40_20_40_pipe_line", 100, 256,
            allocator=allocator.allocate,
            angle_offset_pre_pixel=0.006,
            generator=generator
        )

    def test_angle_segment_pipe_line(self):
        data_set = self.create_real_dataset(filter_method=drive_record_filter_include_all)
        allocator = AngleSegmentRecordAllocator(
            data_set,
            AngleSegment((-1.5, -0.5), 10),  # big sharp left
            AngleSegment((-0.5, -0.25), 14),  # sharp left
            AngleSegment((-0.25, -0.249), 0.5),  # sharp turn left (zero right camera)
            AngleSegment((-0.249, -0.1), 12),  # big turn left
            AngleSegment((-0.1, 0), 13),  # straight left
            AngleSegment((0, 0.001), 1),  # straight zero center camera
            AngleSegment((0.001, 0.1), 13),  # straight right
            AngleSegment((0.1, 0.25), 12),  # big turn right
            AngleSegment((0.25, 0.251), 0.5),  # sharp turn right (zero left camera)
            AngleSegment((0.251, 0.5), 14),  # sharp right
            AngleSegment((0.5, 1.5), 10)  # big sharp right
        )

        # a pipe line with shift -> flip -> brightness -> shadow augment processes
        generator = pipe_line_generators(
            # shift_image_generator(angle_offset_pre_pixel=0.002),
            flip_random_generator,
            brightness_image_generator(0.35),
            shadow_generator
        )

        self._angle_distribution(
            "angle_distribution_generator_angle_segment", 100, 256,
            allocator=allocator.allocate,
            generator=generator
        )

    def test_angle_segment_shift_image(self):
        data_set = self.create_real_dataset(filter_method=drive_record_filter_include_all)
        allocator = AngleSegmentRecordAllocator(
            data_set,
            AngleSegment((-1.5, -0.5), 10),  # big sharp left
            AngleSegment((-0.5, -0.25), 14),  # sharp left
            AngleSegment((-0.25, -0.249), 1),  # sharp turn left (zero right camera)
            AngleSegment((-0.249, -0.1), 12),  # big turn left
            AngleSegment((-0.1, 0), 12),  # straight left
            AngleSegment((0, 0.001), 2),  # straight zero center camera
            AngleSegment((0.001, 0.1), 12),  # straight right
            AngleSegment((0.1, 0.25), 12),  # big turn right
            AngleSegment((0.25, 0.251), 1),  # sharp turn right (zero left camera)
            AngleSegment((0.251, 0.5), 14),  # sharp right
            AngleSegment((0.5, 1.5), 10)  # big sharp right
        )

        # a pipe line with shift -> flip -> brightness -> shadow augment processes
        generator = pipe_line_generators(
            shift_image_generator(angle_offset_pre_pixel=0.0015),
            flip_random_generator,
            brightness_image_generator(0.35),
            shadow_generator
        )

        self._angle_distribution(
            "angle_distribution_generator_angle_segment_shift_image", 100, 256,
            allocator=allocator.allocate,
            generator=generator
        )

    def _angle_distribution(
            self, name, batches, batch_size, allocator,
            angle_offset_pre_pixel=0.002, generator=None
    ):

        if generator is None:
            generator = pipe_line_random_generators(
                image_itself,
                shift_image_generator(angle_offset_pre_pixel=angle_offset_pre_pixel),
                flip_generator
            )
        data_generator = DataGenerator(allocator, generator)
        angles = np.array([])
        for index in range(batches):
            print("batch {} / {}".format(index, batches))
            _, _angles = next(data_generator.generate(batch_size=batch_size))
            angles = np.append(angles, _angles)

        plt = Plot.angle_distribution(angles)
        plt.savefig("angle/{}.jpg".format(name))
