"""Helpers for building USD articulation trees with physics joints."""


def _create_box(stage, path, size, mass):
    from pxr import Gf, UsdGeom, UsdPhysics

    xf = UsdGeom.Cube.Define(stage, path)
    xf.GetSizeAttr().Set(1.0)
    xf_prim = xf.GetPrim()
    UsdGeom.Xformable(xf_prim).AddScaleOp().Set(Gf.Vec3f(*size))
    mass_api = UsdPhysics.MassAPI.Apply(xf_prim)
    mass_api.GetMassAttr().Set(mass)
    return xf_prim


def _create_fixed_joint(stage, path, body0, body1, local_pos=None):
    from pxr import Gf, Sdf, UsdPhysics

    if local_pos is None:
        local_pos = Gf.Vec3f(0, 0, 0)
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.GetLocalPos0Attr().Set(local_pos)
    return joint


def _create_prismatic_joint(
    stage, path, body0, body1, axis="Z", lower_limit=0.0, upper_limit=1.0, drive_stiffness=1e6, drive_damping=1e4
):
    from pxr import Sdf, UsdPhysics

    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.GetAxisAttr().Set(axis)
    joint.GetLowerLimitAttr().Set(lower_limit)
    joint.GetUpperLimitAttr().Set(upper_limit)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
    drive.GetStiffnessAttr().Set(drive_stiffness)
    drive.GetDampingAttr().Set(drive_damping)
    return joint


def build_forklift_articulation(output_path="robot.usd"):
    """Create a minimal forklift USD with chassis, mast, and lift joint."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(output_path)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    base = _create_box(stage, "/Robot/base", size=(2.5, 1.2, 0.4), mass=2000.0)
    UsdPhysics.RigidBodyAPI.Apply(base)

    mast_base = _create_box(stage, "/Robot/base/MastBase", size=(0.1, 1.0, 2.0), mass=200.0)
    UsdPhysics.RigidBodyAPI.Apply(mast_base)
    _create_fixed_joint(
        stage,
        "/Robot/base/MastBase/FixedJoint",
        body0="/Robot/base",
        body1="/Robot/base/MastBase",
        local_pos=Gf.Vec3f(1.3, 0, 0.3),
    )

    inner_mast = _create_box(stage, "/Robot/base/MastBase/InnerMast", size=(0.08, 0.9, 1.8), mass=100.0)
    UsdPhysics.RigidBodyAPI.Apply(inner_mast)
    _create_prismatic_joint(
        stage,
        "/Robot/base/MastBase/InnerMast/LiftJoint",
        body0="/Robot/base/MastBase",
        body1="/Robot/base/MastBase/InnerMast",
        axis="Z",
        lower_limit=0.0,
        upper_limit=3.0,
        drive_stiffness=1e6,
        drive_damping=1e4,
    )

    stage.GetRootLayer().Save()
    return stage
