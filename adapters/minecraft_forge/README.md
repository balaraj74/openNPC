# OpenNPC Minecraft Forge Integration

A fully compiled, ready-to-use Minecraft Forge Mod targeting **Minecraft 1.20.1** that dynamically hooks into vanilla Pathfinder entities (e.g., Zombies, Skeletons, Pigs, Cows) and gives them **AI-driven decision making** using the OpenNPC engine!

---

## ⚡ Quick Start: Ready to Play

We have pre-compiled the Forge mod for you! You can drop it directly into TLauncher and start playing with smart AI-driven entities immediately.

### Step 1: Copy the Mod to TLauncher

1. Locate the built mod jar:
   `/home/balaraj/openNPC/adapters/minecraft_forge/opennpc-1.0.0.jar`
2. Copy this file into your Minecraft `mods` folder.
   * On Linux (default TLauncher path): `~/.minecraft/mods/`
   * If the `mods` folder does not exist, create a folder named `mods` inside `~/.minecraft/`.

### Step 2: Start the OpenNPC AI Engine Server

Open a new terminal window, navigate to the openNPC repository root, and run:
```bash
python3 main.py --port 8787
```
*(This starts the OpenNPC high-performance HTTP decision API server at `http://127.0.0.1:8787`)*

### Step 3: Run Minecraft via TLauncher

1. Open TLauncher.
2. In the version dropdown list (bottom right), select **Forge 1.20.1** (or **ForgeOptiFine 1.20.1**).
3. Click **Enter the game** (or **Install** if you haven't played this version yet).
4. Create or load a single-player world.

### Step 4: Watch the AI in Action!

As soon as a Zombie, Pig, Skeleton, or Cow spawns in your vicinity:
1. The Forge mod automatically intercepts its spawn event.
2. It injects the high-priority `OpenNPCGoal` into its AI selector.
3. Every second, the entity gathers its status (health, distance to closest player, position, UUID) and sends an asynchronous request to your local OpenNPC python server.
4. The OpenNPC engine runs it through the ONNX-optimized deep neural network policy (or Redis memory lookup) and returns a command (`attack`, `retreat`, `patrol`, `jump`, `idle`).
5. The Forge mod safely executes the action on the main game thread.
6. You will see real-time interaction logs in both the terminal running `main.py` and the Minecraft server console!

---

## 🛠️ Mod Architecture

If you want to view, modify, or extend the source code of the mod, it is organized in the `project/` directory:

* **[OpenNPCMod.java](project/src/main/java/com/opennpc/mod/OpenNPCMod.java)**:
  * Registers to `EntityJoinLevelEvent` on the Forge event bus.
  * Dynamically hooks into standard `PathfinderMob` entities.
* **[OpenNPCGoal.java](project/src/main/java/com/opennpc/mod/OpenNPCGoal.java)**:
  * Extends Minecraft's `Goal` class.
  * Periodically serializes entity states (health, relative coordinates, target player distance) into a standardized JSON payload.
  * Maps received actions (`attack`, `retreat`, `patrol`, `jump`, `idle`) to Forge AI navigation controls.
* **[OpenNPCClient.java](project/src/main/java/com/opennpc/mod/OpenNPCClient.java)**:
  * Custom asynchronous HTTP Client using Java 17 `java.net.http.HttpClient` to communicate with the Python server on a background threadpool to avoid lag spikes.

---

## 🏗️ Rebuilding from Source

If you edit the Java source files and want to rebuild the `.jar` package, run:
```bash
cd /home/balaraj/openNPC/adapters/minecraft_forge/project
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
./gradlew build --no-daemon
```
The compiled jar will be created at `build/libs/opennpc-1.0.0.jar`.
