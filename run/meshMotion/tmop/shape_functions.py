import torch
from .registries import register_shape

@register_shape
def barycentric_coordinates(p, vertices=None):
    """
    Barycentric coordinates on unit triangle (0,0),(1,0),(0,1).
    """
    return torch.stack([1-p[...,0]-p[...,1], p[...,0], p[...,1]],dim=-1)

@register_shape
def laplace_coordinates(p, vertices):
    device = vertices.device
    dtype = vertices.dtype

    n_sides = vertices.shape[0]

    triangles = torch.stack((
        torch.full((n_sides,), n_sides, device=device),
        torch.arange(n_sides, device=device),
        (torch.arange(n_sides, device=device) + 1) % n_sides
    ), dim=1)

    cs_list = []
    stack_pts = torch.vstack([vertices,p[None,:]])
    for i,tri in enumerate(triangles):
        pts = stack_pts[tri]

        # shamelessly yoinked from chatgpt
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]

        r0 = x0*x0 + y0*y0
        r1 = x1*x1 + y1*y1
        r2 = x2*x2 + y2*y2

        D = 2 * torch.linalg.det(
            torch.stack((
                torch.stack((x0, y0, torch.ones((), device=device, dtype=dtype))),
                torch.stack((x1, y1, torch.ones((), device=device, dtype=dtype))),
                torch.stack((x2, y2, torch.ones((), device=device, dtype=dtype))),
            ))
        )

        Ux = torch.linalg.det(
            torch.stack((
                torch.stack((r0, y0, torch.ones((), device=device, dtype=dtype))),
                torch.stack((r1, y1, torch.ones((), device=device, dtype=dtype))),
                torch.stack((r2, y2, torch.ones((), device=device, dtype=dtype))),
            ))
        ) / D

        Uy = torch.linalg.det(
            torch.stack((
                torch.stack((x0, r0, torch.ones((), device=device, dtype=dtype))),
                torch.stack((x1, r1, torch.ones((), device=device, dtype=dtype))),
                torch.stack((x2, r2, torch.ones((), device=device, dtype=dtype))),
            ))
        ) / D

        cs_list.append(torch.stack((Ux, Uy)))

    cs = torch.stack(cs_list)


    s = torch.stack( [ torch.linalg.norm( cs[i] - cs[i-1] ) for i in range(n_sides) ] )
    h = torch.stack( [ torch.linalg.norm( vertices[i] - p ) for i in range(n_sides) ] )
    alpha = s/h
    return alpha/alpha.sum()

@register_shape
def mean_value_coordinates(p, vertices):
    d = vertices - p.unsqueeze(0)
    r = torch.linalg.norm(d,dim=1)
    e = d/r.unsqueeze(-1)
    e_p = torch.roll(e,-1,0)

    cos = torch.linalg.vecdot(e,e_p) 
    sin = torch.linalg.det( torch.stack((e,e_p),dim=-1) ) 
    tan = sin/(1+cos)

    tan_m =  torch.roll(tan,1,0)

    w = (tan_m + tan)/r
    return w/w.sum()


@register_shape
def mean_value_coordiantes_global(p,vertices):
    eps = 1e-12
    n = vertices.shape[0]
    d = vertices - p.unsqueeze(0)
    r = torch.linalg.norm(d,dim=1)

    dp = torch.roll(d,1,dims=0)
    dn = torch.roll(d,-1,dims=0)

    rp = torch.roll(r,1,dims=0)
    rn = torch.roll(r,-1,dims=0)

    ws = []
    for i in range(n):
        val = rp[i] * rn[i] - dp[i].dot(dn[i])
        # a = torch.sqrt(torch.clamp(val, min = 0.0 ))
        a = torch.sqrt(val + eps)
        b = 1.0
        for j in range(n):
            if (j == (i-1)%n) or (j == i):
                continue
            val = r[j]*rn[j] + d[j].dot(dn[j])
            # b *= torch.sqrt(torch.clamp(val,min=0.0))
            b *= torch.sqrt(val + eps)
        ws.append(a*b)

    w = torch.stack(ws)
    return w/w.sum()


def _angle(a, b):
    sin = a[...,0] * b[...,1] - a[...,1] * b[...,0]
    cos = a[...,0] * b[...,0] + a[...,1] * b[...,1]
    return torch.atan2(sin, cos)

# this version seems the most stable for mvc.
# Implementation in the paper is different though
@register_shape
def stable_mean_value_coordinates(p,vertices):
    n = vertices.shape[0]
    d = vertices - p.unsqueeze(0)
    r = torch.linalg.norm(d,dim=1)

    dp = torch.roll(d,1,dims=0)
    dn = torch.roll(d,-1,dims=0)

    rp = torch.roll(r,1,dims=0)
    rn = torch.roll(r,-1,dims=0)

    alphap = _angle(dp,d)
    alphan = _angle(d,dn)

    # alphapn = angle(dp,dn) 
    # alphapn *= ( 2 * ((alphapn * (alphap+alphan)) > 0) - 1 )

    ws = []
    for i in range(n):
        a = torch.sin(0.5*(alphap[i] + alphan[i])) * rp[i]
        b = 1.0
        for j in range(n):
            if (j == (i-1)%n) or (j == i):
                continue
            b *= r[j] * torch.cos(alphan[j]*0.5)
        ws.append(a*b)

    w = torch.stack(ws)
    return w/w.sum()
