import sys
import Transform_Moodboard
import RBM_Train
import Generate_Grid

if __name__ == '__main__':
    style = None
    info_input = "Escoja el estilo deseado: \n 1.- Popular Modern \n 2.- Popular Traditional \n 3.- " \
                 "Scandinavian Modern \n 4.- Scandinavian Traditional  \n 5.- Salir \n"
    while not style:
        try:
            style = int(input(info_input))
        except ValueError:
            print('Invalid input data!!')

    if style == 1:
        transform_grids = input("Transformar moodboards para entrenar la màquina? (S/N) \n")
        if transform_grids.lower() == 's':
            Transform_Moodboard.main('Popular_Modern.json')
            print("Has generado nuevos datos, es necesario entrenar a la màquina!!")
        train_machine = input("Quieres entrenar a la màquina? (S/N) \n")
        if train_machine.lower() == "s":
            RBM_Train.main("Popular_Modern")
        generate_grid = None
        while not generate_grid:
            try:
                generate_grid = int(input("Quantos moodboards nuevos quieres generar? \n"))
                Generate_Grid.main(generate_grid, "Popular_Modern")
            except ValueError:
                print("Formato Incorrecto!")
                sys.exit()

    elif style == 2:
        print("Popular Traditional")
    elif style == 3:
        print("Scandinavian Modern")
    elif style == 4:
        print("Scandinavian Traditional")
    else:
        sys.exit()
