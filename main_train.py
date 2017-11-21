from data_load import DriveDataSet, DataGenerator, drive_record_filter_include_all, AngleTypeWithZeroRecordAllocator, \
    AngleSegmentRecordAllocator, AngleSegment, RecordRandomAllocator
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator, \
    flip_random_generator
from trainer import Trainer
from model import nvidia, nvidia_with_regularizer
from donkeycar.parts.datastore import TubHandler, TubGroup


def is_osx():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        return False
    elif _platform == "darwin":
        return True
    elif _platform == "win32":
        return False


use_multi_process = not is_osx()


def create_real_dataset(filter_method):
    tubgroup = TubGroup("data/log_1,data/log_2,data/log_3,data/log_w_6,data/log_w_7")

    print("splitting train / validation 0.9/0.1")

    train_df = tubgroup.df.sample(frac=0.9, random_state=200)
    val_df = tubgroup.df.drop(train_df.index)

    train_data_set = DriveDataSet.from_tubgroup(
        train_df,
        filter_method=filter_method,
        fake_image=False
    )
    val_data_set = DriveDataSet.from_tubgroup(
        val_df,
        filter_method=filter_method,
        fake_image=False
    )

    print("dataset created")

    return train_data_set, val_data_set


def segment_normal_distribution_shift_flip_brightness_shadow_reg():
    data_set_train, data_set_val = create_real_dataset(filter_method=drive_record_filter_include_all)

    # fine tune every part of training data so that make it meat std distrubtion
    allocator_train = AngleSegmentRecordAllocator(
        data_set_train,
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
    allocator_val = AngleSegmentRecordAllocator(
        data_set_val,
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
    data_generator_train = DataGenerator(allocator_train.allocate, pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.0015),
        flip_random_generator,
        brightness_image_generator(0.35),
        shadow_generator
    ))
    data_generator_val = DataGenerator(allocator_val.allocate, pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.0015),
        flip_random_generator,
        brightness_image_generator(0.35),
        shadow_generator
    ))
    model = nvidia_with_regularizer(input_shape=data_set_train.output_shape(), dropout=0.2)
    Trainer(model, learning_rate=0.0001, epoch=45, multi_process=use_multi_process,
            custom_name="mypilot").fit_generator(
        data_generator_train.generate(batch_size=256),
        data_generator_val.generate(batch_size=256)
    )


if __name__ == "__main__":
    segment_normal_distribution_shift_flip_brightness_shadow_reg()