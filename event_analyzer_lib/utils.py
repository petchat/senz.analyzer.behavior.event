def check_2D_list(D_list):
    '''Return True if the_list is 2-dimension else false
    '''
    if isinstance(D_list, list):
        if len(D_list) > 0 and isinstance(D_list[0], list):
            return True

    return False

if __name__ == '__main__':
    print(check_2D_list([1,2,3]))
    print(check_2D_list([[1,2,3], [4,5,6]]))
    print(check_2D_list([[], []]))
    print(check_2D_list([]))
