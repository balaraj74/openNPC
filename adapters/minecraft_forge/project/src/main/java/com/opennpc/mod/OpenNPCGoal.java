package com.opennpc.mod;

import com.mojang.logging.LogUtils;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.PathfinderMob;
import net.minecraft.world.entity.ai.goal.Goal;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import java.util.EnumSet;

/**
 * AI Goal that replaces vanilla behavior with decisions from the OpenNPC server.
 * Handles the full combat loop including actual melee damage dealing.
 */
public class OpenNPCGoal extends Goal {
    private static final Logger LOGGER = LogUtils.getLogger();
    private final PathfinderMob mob;
    private final OpenNPCClient client;

    // Decision polling
    private int decisionCooldown = 0;
    private boolean isQuerying = false;
    private String currentAction = "patrol";

    // Melee attack timing (vanilla zombie attacks every 20 ticks = 1 second)
    private int attackCooldown = 0;
    private static final int ATTACK_INTERVAL = 20;
    private static final double MELEE_REACH = 2.0;

    public OpenNPCGoal(PathfinderMob mob, OpenNPCClient client) {
        this.mob = mob;
        this.client = client;
        // Control all three flags so vanilla goals don't interfere
        this.setFlags(EnumSet.of(Flag.MOVE, Flag.LOOK, Flag.JUMP));
    }

    @Override
    public boolean canUse() {
        return mob.isAlive();
    }

    @Override
    public boolean canContinueToUse() {
        return mob.isAlive();
    }

    @Override
    public void tick() {
        // Count down attack cooldown every tick
        if (attackCooldown > 0) {
            attackCooldown--;
        }

        // Count down decision cooldown
        if (decisionCooldown > 0) {
            decisionCooldown--;
        }

        // Query the AI server periodically
        if (decisionCooldown <= 0 && !isQuerying) {
            queryAI();
        }

        // Execute the current action every tick
        executeAction();
    }

    private void queryAI() {
        isQuerying = true;

        Player nearestPlayer = mob.level().getNearestPlayer(mob, 32.0D);
        double distance = nearestPlayer != null ? mob.distanceTo(nearestPlayer) : -1;
        float playerHealth = nearestPlayer != null ? nearestPlayer.getHealth() : -1;
        boolean hasTarget = nearestPlayer != null && distance >= 0 && distance <= 24.0;

        // Compute threat level based on proximity
        float threatLevel = 0.1f;
        if (nearestPlayer != null) {
            if (distance <= 2.0) threatLevel = 0.95f;
            else if (distance <= 4.0) threatLevel = 0.8f;
            else if (distance <= 8.0) threatLevel = 0.6f;
            else if (distance <= 16.0) threatLevel = 0.4f;
            else threatLevel = 0.2f;
        }

        // Build config — aggressive enemy personality
        String configJson = String.format(
            "{\"agent_id\":\"%s\",\"agent_type\":\"enemy\","
            + "\"personality\":{\"aggression\":0.9,\"caution\":0.1,\"risk_tolerance\":0.85,"
            + "\"patience\":0.2,\"curiosity\":0.3,\"sociability\":0.05,\"loyalty\":0.5},"
            + "\"allowed_actions\":[\"patrol\",\"attack\",\"retreat\",\"idle\",\"move\",\"flank\",\"defend\"]}",
            mob.getStringUUID()
        );

        // Build state — include everything the decision engine needs
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append(String.format("\"agent_id\":\"%s\",", mob.getStringUUID()));
        sb.append(String.format("\"health\":%.1f,", mob.getHealth()));
        sb.append(String.format("\"max_health\":%.1f,", mob.getMaxHealth()));
        sb.append(String.format("\"threat_level\":%.2f,", threatLevel));
        sb.append(String.format("\"location\":\"%s\",", mob.blockPosition().toShortString().replace(" ", "")));

        if (hasTarget) {
            sb.append(String.format("\"distance_to_target\":%.2f,", distance));
            sb.append(String.format("\"target_health\":%.1f,", playerHealth));
            sb.append(String.format("\"target_max_health\":%.1f,", nearestPlayer.getMaxHealth()));
            sb.append("\"nearby_entities\":[\"player\"],");
        } else {
            sb.append("\"distance_to_target\":null,");
            sb.append("\"target_health\":null,");
            sb.append("\"nearby_entities\":[],");
        }

        sb.append("\"cover_available\":false,");
        sb.append(String.format("\"last_action\":\"%s\",", currentAction));
        sb.append(String.format("\"tick\":%d", mob.tickCount));
        sb.append("}");

        String stateJson = sb.toString();

        client.decide(configJson, stateJson).thenAccept(response -> {
            mob.level().getServer().execute(() -> {
                String newAction = response.action.toLowerCase();
                if (!newAction.equals(currentAction)) {
                    LOGGER.info("[OpenNPC] {} ({}): {} -> {} | hp={} | dist={} | reason: {}",
                        mob.getType().getDescriptionId(),
                        mob.getStringUUID().substring(0, 8),
                        currentAction, newAction,
                        String.format("%.0f", mob.getHealth()),
                        hasTarget ? String.format("%.1f", distance) : "N/A",
                        response.reason);
                }
                currentAction = newAction;
                isQuerying = false;
                // Faster re-query when in combat (20 ticks = 1s), slower when idle (40 ticks = 2s)
                decisionCooldown = hasTarget ? 20 : 40;
            });
        }).exceptionally(ex -> {
            mob.level().getServer().execute(() -> {
                isQuerying = false;
                decisionCooldown = 40;
                // Default hostile behavior when server is down
                Player p = mob.level().getNearestPlayer(mob, 16.0D);
                if (p != null) {
                    currentAction = "attack";
                }
            });
            return null;
        });
    }

    private void executeAction() {
        Player nearestPlayer = mob.level().getNearestPlayer(mob, 24.0D);

        switch (currentAction) {
            case "attack":
            case "flank":
                if (nearestPlayer != null) {
                    double dist = mob.distanceTo(nearestPlayer);
                    mob.setTarget(nearestPlayer);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    if (dist > MELEE_REACH) {
                        // Chase the player — sprint speed
                        mob.getNavigation().moveTo(nearestPlayer, 1.2D);
                        mob.setSprinting(dist > 5.0);
                    } else {
                        // In melee range — ACTUALLY HIT THE PLAYER
                        mob.getNavigation().stop();
                        if (attackCooldown <= 0) {
                            mob.swing(mob.getUsedItemHand());  // Visual arm swing
                            mob.doHurtTarget(nearestPlayer);   // Deal actual damage
                            attackCooldown = ATTACK_INTERVAL;
                        }
                    }
                } else {
                    // No player nearby, fall back to patrolling
                    doPatrol();
                }
                break;

            case "defend":
                if (nearestPlayer != null) {
                    double dist = mob.distanceTo(nearestPlayer);
                    mob.setTarget(nearestPlayer);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    // Hold ground but attack if player gets close
                    if (dist <= MELEE_REACH && attackCooldown <= 0) {
                        mob.swing(mob.getUsedItemHand());
                        mob.doHurtTarget(nearestPlayer);
                        attackCooldown = ATTACK_INTERVAL;
                    } else if (dist > 6.0D) {
                        // Slowly approach if player is too far
                        mob.getNavigation().moveTo(nearestPlayer, 0.7D);
                    } else {
                        mob.getNavigation().stop();
                    }
                }
                break;

            case "move":
                if (nearestPlayer != null) {
                    mob.setTarget(nearestPlayer);
                    mob.getNavigation().moveTo(nearestPlayer, 1.0D);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    // Still attack if in range while moving
                    if (mob.distanceTo(nearestPlayer) <= MELEE_REACH && attackCooldown <= 0) {
                        mob.swing(mob.getUsedItemHand());
                        mob.doHurtTarget(nearestPlayer);
                        attackCooldown = ATTACK_INTERVAL;
                    }
                }
                break;

            case "retreat":
                if (nearestPlayer != null) {
                    mob.setTarget(null);
                    Vec3 away = mob.position().subtract(nearestPlayer.position()).normalize().scale(10.0D);
                    Vec3 retreatPos = mob.position().add(away);
                    mob.getNavigation().moveTo(retreatPos.x, retreatPos.y, retreatPos.z, 1.4D);
                    mob.setSprinting(true);
                } else {
                    doPatrol();
                }
                break;

            case "patrol":
                doPatrol();
                break;

            case "idle":
            default:
                mob.getNavigation().stop();
                mob.setTarget(null);
                mob.setSprinting(false);
                // Idle zombies still look at nearby players menacingly
                if (nearestPlayer != null && mob.distanceTo(nearestPlayer) <= 8.0D) {
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);
                }
                break;
        }
    }

    private void doPatrol() {
        mob.setTarget(null);
        mob.setSprinting(false);
        if (!mob.getNavigation().isInProgress()) {
            double rx = mob.getX() + (mob.getRandom().nextDouble() - 0.5D) * 12.0D;
            double rz = mob.getZ() + (mob.getRandom().nextDouble() - 0.5D) * 12.0D;
            mob.getNavigation().moveTo(rx, mob.getY(), rz, 0.8D);
        }
    }
}
