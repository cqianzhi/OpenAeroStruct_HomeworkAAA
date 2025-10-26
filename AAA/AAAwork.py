import numpy as np

from openaerostruct.meshing.mesh_generator import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

import openmdao.api as om
from openaerostruct.utils.constants import grav_constant

# Create a dictionary to store options about the surface 
# 几何定义和网格生成
mesh_dict = {"num_y": 7, "wing_type": "CRM", "symmetry": True}

mesh, twist_cp = generate_mesh(mesh_dict)

# 结构属性定义
surface_single_spar = {
    # Wing definition
    "name": "wing_single_spar",  # name of the surface
    "symmetry": True,  # if true, model one half of wing
    # reflected across the plane y = 0
    # "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "fem_model_type": "wingbox",
    "mx": 2,
    "my": 7,
    "thickness_cp": np.array([0.1, 0.2, 0.3]),
    "twist_cp": twist_cp,
    "mesh": mesh,

    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    # "CL0": 0.0,  # CL of the surface at alpha=0
    # "CD0": 0.015,  # CD of the surface at alpha=0
    # Airfoil properties for viscous drag calculation
    # "k_lam": 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    # "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    "with_viscous": True,
    "with_wave": False,  # if true, compute wave drag

    # Structural values are based on aluminum 7075
    "E": 70.0e9,  # [Pa] Young's modulus of the spar
    "G": 30.0e9,  # [Pa] shear modulus of the spar
    "yield": 500.0e6,
    "safety_factor": 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
    "rho": 3.0e3,  # [kg/m^3] material density
    # "fem_origin": 0.35,  # normalized chordwise location of the spar

    # BOOM
    "boom_positions": [0.25],        # 单梁
    # "boom_positions": [0.25, 0.75], # 双梁
    "boom_diameters": [0.1],
    "wall_thickness": 0.01,

    "wing_weight_ratio": 2.0,
    "struct_weight_relief": True,  # True to add the weight of the structure to the loads on the structure AAA REVISED
    "distributed_fuel_weight": False,
    # Constraints
    "exact_failure_constraint": False,  # if false, use KS function
}

surface_double_spar = {
    # Wing definition
    "name": "wing_double_spar",  # name of the surface
    "symmetry": True,  # if true, model one half of wing
    # reflected across the plane y = 0
    # "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "fem_model_type": "wingbox",
    "mx": 2,
    "my": 7,
    "thickness_cp": np.array([0.1, 0.2, 0.3]),
    "twist_cp": twist_cp,
    "mesh": mesh,

    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    # "CL0": 0.0,  # CL of the surface at alpha=0
    # "CD0": 0.015,  # CD of the surface at alpha=0
    # Airfoil properties for viscous drag calculation
    # "k_lam": 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    # "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    "with_viscous": True,
    "with_wave": False,  # if true, compute wave drag

    # Structural values are based on aluminum 7075
    "E": 70.0e9,  # [Pa] Young's modulus of the spar
    "G": 30.0e9,  # [Pa] shear modulus of the spar
    "yield": 500.0e6,
    "safety_factor": 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
    "rho": 3.0e3,  # [kg/m^3] material density
    # "fem_origin": 0.35,  # normalized chordwise location of the spar

    # BOOM
    # "boom_positions": [0.25],        # 单梁
    "boom_positions": [0.25, 0.75], # 双梁
    "boom_diameters": [0.1, 0.1],
    "wall_thickness": 0.01,

    "wing_weight_ratio": 2.0,
    "struct_weight_relief": True,  # True to add the weight of the structure to the loads on the structure AAA REVISED
    "distributed_fuel_weight": False,
    # Constraints
    "exact_failure_constraint": False,  # if false, use KS function
}

# Create the problem and assign the model group
prob = om.Problem()

# Add problem information as an independent variables component
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s") # 飞行速度
indep_var_comp.add_output("alpha", val=3.25, units="deg") # 攻角（Angle of Attack）
indep_var_comp.add_output("Mach_number", val=0.85) # 马赫数（0.84 ≈ 高亚音速巡航）
indep_var_comp.add_output("re", val=1.04e7, units="1/m") # 雷诺数（Reynolds number）
indep_var_comp.add_output("rho", val=0.36, units="kg/m**3") # 空气密度
indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s") # 发动机比推力相关参数
indep_var_comp.add_output("R", val=11.165e6, units="m") # 地球半径
indep_var_comp.add_output("W0", val=175000, units="kg") # 飞机初始重量
indep_var_comp.add_output("speed_of_sound", val=295.4, units="m/s") # 声速
indep_var_comp.add_output("load_factor", val=1.0) # 载荷系数（1.0 = 平飞）
indep_var_comp.add_output("empty_cg", val=np.array([25.6, 0.0, 0.0]), units="m") # 空机重心位置
prob.model.add_subsystem("inputs", indep_var_comp, promotes=["*"])

# group_single_spar = AerostructGeometry(surface=surface_single_spar)
# group_double_spar = AerostructGeometry(surface=surface_double_spar)


# Add tmp_group to the problem with the name of the surface.
# prob.model.add_subsystem("wing_single_spar", group_single_spar)
# prob.model.add_subsystem("wing_double_spar", group_double_spar)

point_name = "AS_point_0"

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=[surface_single_spar, surface_double_spar])

prob.model.add_subsystem(
    point_name,
    AS_point,
    promotes_inputs=["*"],
)
# prob.model.connect("inputs.v", "AS_point_0.v") 
"""
# 连接独立变量到 AS_point
prob.model.connect("inputs.v", point_name + ".v")
prob.model.connect("inputs.alpha", point_name + ".alpha")
prob.model.connect("inputs.Mach_number", point_name + ".Mach_number")
prob.model.connect("inputs.re", point_name + ".re")
prob.model.connect("inputs.rho", point_name + ".rho")
prob.model.connect("inputs.CT", point_name + ".CT")
prob.model.connect("inputs.R", point_name + ".R")
prob.model.connect("inputs.W0", point_name + ".W0")
prob.model.connect("inputs.speed_of_sound", point_name + ".speed_of_sound")
prob.model.connect("inputs.empty_cg", point_name + ".empty_cg")
prob.model.connect("inputs.load_factor", point_name + ".load_factor")
"""
"""
com_name = point_name + "." + name + "_perf"
prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

# Connect aerodyamic mesh to coupled group mesh
prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")

# Connect performance calculation variables
prob.model.connect(name + ".radius", com_name + ".radius")
prob.model.connect(name + ".thickness", com_name + ".thickness")
prob.model.connect(name + ".nodes", com_name + ".nodes")
prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")
"""

# 设置优化器
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

# 添加记录器
recorder = om.SqliteRecorder("aerostruct.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["record_derivatives"] = True
prob.driver.recording_options["includes"] = ["*"]

# Setup problem and add design variables, constraint, and objective
# 扭转控制点
prob.model.add_design_var(point_name + ".wing_single_spar.twist_cp", lower=-5., upper=10.)
prob.model.add_design_var(point_name + ".wing_double_spar.twist_cp", lower=-5., upper=10.)
# 厚度控制点
prob.model.add_design_var(point_name + ".wing_single_spar.thickness_cp",
                          lower=0.01, upper=0.1, scaler=1e2)
prob.model.add_design_var(point_name + ".wing_double_spar.thickness_cp",
                          lower=0.01, upper=0.1, scaler=1e2)
# 结构失效约束
prob.model.add_constraint(point_name + ".wing_single_spar.failure", upper=0.0)
prob.model.add_constraint(point_name + ".wing_double_spar.failure", upper=0.0)
# prob.model.add_constraint(point_name + ".wing_single_spar.thickness_intersects", upper=0.0)
# prob.model.add_constraint(point_name + ".wing_double_spar.thickness_intersects", upper=0.0)

# Add design variables, constraisnt, and objective on the problem
prob.model.add_design_var("alpha", lower=-5.0, upper=10.0)
prob.model.add_constraint(point_name + ".L_equals_W", equals=0.0) # 配平约束：升力 = 重力
prob.model.add_objective(point_name + ".fuelburn", scaler=1e-5) # 最小化燃油消耗


# Set up the problem
prob.setup(check=True)

# Only run analysis
# prob.run_model()

# Run optimization
prob.run_driver()

print("Single Spar CL:", prob[point_name + ".wing_single_spar.CL"])
print("Double Spar CL:", prob[point_name + ".wing_double_spar.CL"])
print("Total CL:", prob[point_name + ".total_perf.CL"])
print("Total CD:", prob[point_name + ".total_perf.CD"])
print("Fuel Burn:", prob[point_name + ".fuelburn"])
"""
print()
print("CL:", prob["AS_point_0.wing_perf.CL"])
print("CD:", prob["AS_point_0.wing_perf.CD"])
"""
