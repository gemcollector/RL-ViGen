from dm_control.utils import rewards



def main():
    for i in [0.3, 0.4]:
        print(rewards.tolerance(i, bounds=(0, 0.01), margin=0.02))



if __name__ == '__main__':
    main()