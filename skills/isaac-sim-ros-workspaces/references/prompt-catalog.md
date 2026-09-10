# Prompt Catalog

Use this catalog to distinguish this skill's workflows and neighboring skills. Prompts are
examples, not exact-string matching rules.

| ID | Example prompt | Expected route | Entrypoint |
|---|---|---|---|
| native-01 | Build IsaacSim-ros_workspaces on my Ubuntu 24.04 machine; Jazzy is installed. | Native Jazzy | `setup_ros_workspace.sh --method native` |
| native-02 | Clone the Isaac Sim Humble workspace and run colcon using my existing ROS setup. | Native Humble | `setup_ros_workspace.sh --method native --distro humble` |
| auto-01 | Set up IsaacSim-ros_workspaces on Linux. | Native if matching ROS exists; otherwise Docker | `setup_ros_workspace.sh` |
| auto-02 | Check whether ROS is installed, then build the workspace. | Let the script check `/opt/ros/<distro>/setup.bash`; do not infer absence from `which ros2` | `setup_ros_workspace.sh` |
| docker-01 | Build the Jazzy Isaac Sim ROS workspace in Docker. | Prebuilt-image Docker | `setup_ros_workspace.sh --method docker --distro jazzy` |
| docker-02 | ROS 2 is not installed; build the tutorial workspace for me. | Prebuilt-image Docker | `setup_ros_workspace.sh --method docker` |
| docker-03 | Containerize the Humble workspace build on Ubuntu 22.04. | Prebuilt-image Docker | `setup_ros_workspace.sh --method docker --distro humble` |
| docker-04 | Build humble_ws on Ubuntu 24.04. | Docker Humble (native rejected) | `setup_ros_workspace.sh --method docker --distro humble` |
| docker-05 | Build jazzy_ws on Ubuntu 22.04. | Docker Jazzy (native rejected) | `setup_ros_workspace.sh --method docker --distro jazzy` |
| custom-01 | Build the custom ROS messages so Isaac Sim Python can import them. | Custom Python 3.12 | `setup_ros_workspace.sh --method custom` |
| custom-02 | Run build_ros.sh for Jazzy on Ubuntu 22.04. | Custom Python 3.12 | `setup_ros_workspace.sh --method custom --distro jazzy` |
| custom-03 | Build my custom interfaces in Docker for Isaac Sim's Python 3.12. | Custom wins over generic Docker | `setup_ros_workspace.sh --method custom` |
| pixi-win-01 | Set up the Isaac Sim ROS workspace on Windows 11. | Windows Jazzy Pixi | `powershell.exe -File .../setup_pixi_workspace.ps1` |
| pixi-win-02 | Install Pixi and Git, then build jazzy_ws with MSVC. | Windows Jazzy Pixi | `powershell.exe -File .../setup_pixi_workspace.ps1` |
| pixi-win-03 | Setup sim workspace on this Windows system; check MSVC first. | Run the `.ps1` gates; do not invent `${env:ProgramFiles(x86)}` probes | `powershell.exe -File .../setup_pixi_workspace.ps1` |
| pixi-linux-01 | Explicitly use Pixi to build the ROS workspace on Ubuntu 24.04. | Linux Pixi Jazzy | `setup_ros_workspace.sh --method pixi` |
| internal-01 | Clone our internal IsaacSim-ros_workspaces mirror and build natively. | Native with URL override | `setup_ros_workspace.sh --repo-url ...` |
| internal-02 | Build release/6.0 from our private mirror on Windows. | Windows Pixi with URL/branch overrides | `setup_pixi_workspace.ps1 -RepoUrl ... -Branch ...` |
| existing-01 | The workspace repo is already cloned; set it up for me. | Ask reuse or clean before running; recommend reuse | No clean flag after reuse confirmation |
| existing-02 | Delete my existing checkout and start clean. | Confirm destructive intent is explicit, then clean-reclone | `--clean-reclone --yes-i-know` or `-CleanReclone -YesIKnow` |

## Ambiguous prompts

| Prompt | Resolution |
|---|---|
| Build it with Docker. | Choose prebuilt-image Docker unless prior context established custom interfaces/Python 3.12. |
| Build the custom messages in Docker. | Choose Custom because custom-interface intent outranks generic Docker wording. |
| Use the repository Dockerfile. | Ask whether the user means `build_ros.sh`; choose Custom when confirmed. |
| Build the workspace. | On Linux auto-select native if matching ROS exists, otherwise Docker. On Windows choose Pixi. |
| `which ros2` returns nothing, but `/opt/ros/<distro>/setup.bash` exists. | ROS is installed in an unsourced shell; use the native method and let the script source it. |
| Build the workspace and the target checkout already exists. | Ask whether to reuse it or create a clean clone. Explain that cleaning deletes local and untracked changes. |
| Reuse an existing checkout whose branch differs from the requested branch. | Ask whether to use its current branch, choose another path, or clean-reclone. Never switch or delete automatically. |

Ask one focused question only when the requested output matters and cannot be inferred:
“Must these generated interfaces load inside Isaac Sim’s Python 3.12 runtime?” A yes answer means
Custom; a no answer means the prebuilt-image Docker workspace build.

## Negative and unsupported prompts

| Example prompt | Expected behavior |
|---|---|
| Publish camera topics with the Isaac Sim ROS bridge. | Route to `isaac-sim-ros2-bridge`; do not invoke this skill. |
| Configure Nav2 namespaces for two robots. | Route to `isaac-sim-ros2-bridge`. |
| Set up a workspace. | Do not invoke this skill; no Isaac Sim ROS workspace context is present. |
| Install ROS 2 Jazzy with apt. | Out of scope; this skill does not install public ROS. |
| Build Humble on Ubuntu 24.04. | Use Docker with Humble image; do not attempt native. |
| Build Jazzy on Ubuntu 22.04. | Use Docker with Jazzy image; do not attempt native. |
| Build Humble natively on Ubuntu 24.04. | Stop: native requires matching host-default distro; offer Docker. |
| Build Humble with Pixi on Windows. | Stop: Windows Pixi supports Jazzy only. |
| Set this up through WSL2. | Stop: deprecated and excluded. |
| Install Docker for me. | Stop: Docker CE is a prerequisite, not installed by this skill. |

## Expected failure prompts

| Environment/request | Expected response |
|---|---|
| Docker selected but CLI missing | Hard fail before clone/build with Docker CE remediation. |
| Docker daemon unavailable | Hard fail with daemon-start remediation. |
| Docker permission denied | Hard fail with docker-group and re-login/`newgrp docker` remediation. |
| Native selected but ROS setup absent | Fail and offer this skill's Docker method only. |
| Native tools already installed in a non-interactive session | Skip apt and sudo; continue with rosdep and colcon. |
| Native rosdep hits whiptail or PEP 668 pip errors | Script owns `DEBIAN_FRONTEND=noninteractive`, `NEEDRESTART_MODE=a`, and `PIP_BREAK_SYSTEM_PACKAGES=1`; do not ask the user to export them manually. |
| Native tool missing and sudo cannot prompt | Fail before clone/build and print exact install or `sudo -v` remediation. |
| Linux Pixi selected but `pixi.toml` absent | Fail and identify the missing file; do not generate an undocumented environment. |
| Existing target is not a Git checkout | Stop and request another path; never remove unrelated content. |
| Clean-reclone not explicitly confirmed | Do not pass `--clean-reclone`/`-CleanReclone` or `--yes-i-know`/`-YesIKnow`. |
| Windows MSVC missing | Hard fail with VS 2022 C++ workload prerequisite from the `.ps1`, not from ad-hoc agent probes. |
| Agent MSVC probe used `'${env:ProgramFiles(x86)}\...'` | Treat as invalid diagnostic; rerun `setup_pixi_workspace.ps1 -File` or use the literal `C:\Program Files (x86)\...` path. |
