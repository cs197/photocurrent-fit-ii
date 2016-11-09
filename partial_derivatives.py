
# For numerically taking a derivative
EPSILON = 0.001


def make_d_d_eta(ro_sham_bo):

    def d_d_eta(eta, v0):
        return (ro_sham_bo(eta + EPSILON, v0) - ro_sham_bo(eta, v0)) / EPSILON

    return d_d_eta


def make_d_d_v0(ro_sham_bo):

    def d_d_V0(eta, v0):
        return (ro_sham_bo(eta, v0 + EPSILON) - ro_sham_bo(eta, v0)) / EPSILON

    return d_d_V0


def make_d2_d_eta2(ro_sham_bo):

    return make_d_d_eta(make_d_d_eta(ro_sham_bo))


def make_d2_d_x02(ro_sham_bo):

    return make_d_d_v0(make_d_d_v0(ro_sham_bo))


def make_d2_d_eta_d_v0(ro_sham_bo):

    return make_d_d_eta(make_d_d_v0(ro_sham_bo))
