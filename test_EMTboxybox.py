import numpy as np

from EMT_boxybox import *


def test1():
    ##### TEST 1: run the boxy box (no guarantee on the right hand side)
    print('##### TEST 1: run the boxy box (no guarantee on the right hand side)')
    failed_iters = 0
    bistability = 0
    iterval_boxybox = 15.
    iters = 10

    not_coplanar = 0

    for j in range(iters):
        par = np.random.random(size=(12, 3))
        gamma = np.random.random(size=(6))

        success, xminus, xplus, remainder = boxy_box_from_pars(iterval_boxybox, par, gamma)

        if not success:
            failed_iters += 1
            # plt.semilogy(remainder[0:40])
        else:

            if np.linalg.norm(xplus - xminus) > 0.1:
                bistability += 1
    # plt.show()

    print('No corner equilibria ', failed_iters, 'times out of', iters)
    print('Bistability found ', bistability, 'times out of', iters)
    print('not_coplanar found ', not_coplanar, 'times out of', iters)
    assert bistability > 0


def test2():
    ##### TEST 2: compare boxy box and EMT (test guarantee on the right hand side)
    print('##### TEST 2: compare boxy box and EMT (test guarantee on the right hand side)')
    # set EMT-specific elements

    xminus, xplus = 0, 0
    test_success = False
    norms = 0
    while np.linalg.norm(xminus - xplus) < 10 ** -1 or not test_success:
        par_NDMA = np.abs(np.random.random(42))
        hill = 3.2
        n, par, gamma = NDMApars_to_boxyboxpars(hill, par_NDMA)
        # print(hill, par_NDMA)
        # print(n, par, gamma)
        test_success, xminus, xplus, remainder = boxy_box_from_pars(n, par, gamma)
        all_corners = corners_of_box(xminus, xplus)
        norms = [np.linalg.norm(f(all_corners[i, :], hill, par_NDMA)) for i in range(np.size(all_corners, 1))]
    assert test_success

    print('norms = ', norms)

    F_box = F_func(n, par, gamma)
    print('F_box =', F_box(all_corners[0, :]), '\nF_ndma =', f(all_corners[0, :], hill, par_NDMA))
    print('minimal_norm = ', np.min(norms))
    assert np.min(norms) < 0.1

    ##### TEST 3: find equilibria from boxy box results
    print('##### TEST 3: find equilibria from boxy box results')
    eqs = f.local_equilibrium_search(all_corners, hill, par_NDMA)
    # print(eqs)

    eqs = f.remove_doubles(eqs, hill, par_NDMA)
    print(eqs)
    assert eqs

    # small test: are equilibria on the corners?
    # distance_to_nearest_corner = [np.linalg.norm(eqs[i, :]-all_corners[j, :]) for i in range(2) for j in range(64)]
    # print(distance_to_nearest_corner)

    ##### TEST 4: find change in number of equilibria
    print('##### TEST 4: find change in number of equilibria')

    def count_eqs(all_corners, hill, par_NDMA):
        eqs = f.local_equilibrium_search(all_corners, hill, par_NDMA)
        eqs = f.remove_doubles(eqs, hill, par_NDMA)
        number_of_eqs = np.size(eqs[:, 0])
        return eqs, number_of_eqs

    old_eqs, starting_n_eqs = count_eqs(all_corners, hill, par_NDMA)
    print('starting number of equilibria = ', starting_n_eqs)
    for hill_iter in np.linspace(hill, 1, 50):
        success, xminus, xplus, remainder = boxy_box_from_pars(hill_iter, par, gamma)
        all_corners = corners_of_box(xminus, xplus)
        eqs, n_eqs = count_eqs(all_corners, hill_iter, par_NDMA)
        if n_eqs < 2:
            break
        old_eqs = eqs
    print('approximate saddle node found at hill = ', hill_iter)

    print('equilibria before and after the bifurcation')
    print(old_eqs, eqs)
    assert old_eqs != eqs

    def select_saddle(old_eqs, new_eqs):
        if np.size(new_eqs[:, 1]) > 1:
            print('Not monostability')
            return None
        distances = [np.linalg.norm(x - new_eqs) for x in old_eqs]
        distance_index = np.argmin(distances)
        old_eqs = np.delete(old_eqs, distance_index, axis=0)
        return old_eqs

    saddle_candidate = select_saddle(old_eqs, eqs)
    print('equilibria undergoing saddle = ', saddle_candidate)
    assert saddle_candidate

    saddle_node_problem = SaddleNode(f)
    par_of_SNbif1 = saddle_node_problem.find_saddle_node(0, hill_iter, par_NDMA, equilibria=saddle_candidate)
    print('found saddle node bifurcation', par_of_SNbif1)
    assert par_of_SNbif1

    ##### TEST 5: smoother code to find saddle nodes
    print('##### TEST 5: smoother code to find saddle nodes')
    # for now, keep the same parameter as earlier
    # par_NDMA = np.abs(np.random.random(42))
    high_hill = 10
    low_hill = 1

    high_hill, par, gamma = NDMApars_to_boxyboxpars(high_hill, par_NDMA)
    success, old_xminus, old_xplus, remainder = boxy_box_from_pars(high_hill, par, gamma, maxiter=300)
    old_hill = high_hill
    degeneracy_coef = np.linalg.norm(old_xminus - old_xplus)
    if is_degenerate(old_xminus, old_xplus):
        print('Parameter is monostable at high hill')
    else:
        for hill_iter in np.linspace(high_hill, low_hill, 50):
            success, xminus, xplus, remainder = boxy_box_from_pars(hill_iter, par, gamma, maxiter=2000)
            if is_degenerate(xminus, xplus, tol=degeneracy_coef / 2):
                break
            old_xminus, old_xplus, old_hill = xminus, xplus, hill_iter
            degeneracy_coef = np.linalg.norm(old_xminus - old_xplus)
    print('approximate saddle node found between hill = ', old_hill, hill_iter)
    print('old bounds = ', old_xminus, old_xplus, '\nnew bounds = ', xminus, xplus)

    new_eq = xminus
    approx_saddle_position = 0 * xminus
    for i in range(np.size(xminus)):
        if np.abs(xminus[i] - old_xminus[i]) > np.abs(xminus[i] - old_xplus[i]):
            approx_saddle_position[i] = old_xminus[i]
        else:
            approx_saddle_position[i] = old_xplus[i]
    print('equilibria undergoing saddle = ', approx_saddle_position)

    saddle_node_problem = SaddleNode(f)
    par_of_SNbif2 = saddle_node_problem.find_saddle_node(0, old_hill, par_NDMA, equilibria=approx_saddle_position)
    print('found saddle node bifurcation', par_of_SNbif2)

    assert par_of_SNbif2

    #### TEST 6: run search function
    print('#### TEST 6: run search function automatically')
    approx_saddle_position, old_hill = approx_saddle_node_with_boxy_box([1, 10], par_NDMA)
    print('aproximate saddle node position = ', approx_saddle_position, '\n approximate hill = ', old_hill)
    saddle_node_problem = SaddleNode(f)
    par_of_SNbif3 = saddle_node_problem.find_saddle_node(0, old_hill[0], par_NDMA, equilibria=approx_saddle_position[0])
    print('found saddle node bifurcation', par_of_SNbif3)
    assert par_of_SNbif3
