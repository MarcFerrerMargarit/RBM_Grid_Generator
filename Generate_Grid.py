import sys
import Utils


if __name__ == '__main__':
    number_grids = 1
    if len(sys.argv) > 1:
        number_grids = int(sys.argv[1])
    output_data = Utils.generateGrid(number_grids)
    # En comptes de mostrar graella transformla mitjançant la api
    print(output_data)
