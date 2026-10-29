parameters = {
    "config version": 2,
    "mesh_name": "mesh.h5",
    "solver settings": {"type": "compressible"},
    "equations": {"type": "euler"},
    "boundary conditions": {
        "farfield": {"zones": [1], "type": "farfield", "condition": "IC_1"},
    },
}
