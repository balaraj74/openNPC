package com.opennpc.mod;

import com.mojang.logging.LogUtils;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.PathfinderMob;
import net.minecraft.world.entity.monster.Monster;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.entity.EntityJoinLevelEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.slf4j.Logger;

@Mod(OpenNPCMod.MODID)
public class OpenNPCMod {
    public static final String MODID = "opennpc";
    private static final Logger LOGGER = LogUtils.getLogger();
    private static final OpenNPCClient CLIENT = new OpenNPCClient();

    public OpenNPCMod() {
        MinecraftForge.EVENT_BUS.register(this);
        LOGGER.info("OpenNPC Mod Initialized — AI-driven hostile mob behavior enabled");
    }

    @SubscribeEvent
    public void onEntityJoin(EntityJoinLevelEvent event) {
        // Only attach to hostile mobs (zombies, skeletons, etc), server-side only
        if (!event.getLevel().isClientSide() && event.getEntity() instanceof Monster mob) {
            // Remove ALL vanilla AI goals so OpenNPC has full control
            mob.goalSelector.removeAllGoals(goal -> true);
            mob.targetSelector.removeAllGoals(goal -> true);

            // Add our AI goal at priority 1 (high priority)
            mob.goalSelector.addGoal(1, new OpenNPCGoal(mob, CLIENT));

            LOGGER.info("[OpenNPC] Attached to {} (UUID: {})",
                mob.getType().getDescriptionId(), mob.getStringUUID().substring(0, 8));
        }
    }
}
