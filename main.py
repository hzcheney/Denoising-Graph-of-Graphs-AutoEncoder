from DVGGALearning import DVGGALearning
from param import para_parser
import datetime


def main():
    args = para_parser()
    args.date_time = "{0}-{1}-{2}".format(
        datetime.datetime.now().month,
        datetime.datetime.now().day,
        datetime.datetime.now().strftime("%H:%M:%S"),
    )
    trainer = DVGGALearning(args)
    print("Start training")
    trainer.fit()
    res = trainer.test()
    print(res)


if __name__ == "__main__":
    main()
