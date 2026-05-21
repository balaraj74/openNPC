package com.opennpc.mod;

import com.mojang.logging.LogUtils;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.PathfinderMob;
import net.minecraft.world.entity.ai.goal.Goal;
import net.minecraft.world.entity.monster.Zombie;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;

import java.util.EnumSet;
import java.util.List;
import java.util.stream.Collectors;

/**
 * AI Goal that replaces vanilla behavior with decisions from the OpenNPC server.
 * Zombies are treated as VILLAIN agents with full strategic planning, memory,
 * pattern tracking, and LLM-generated dialogue barks.
 */
public class OpenNPCGoal extends Goal {
    private static final Logger LOGGER = LogUtils.getLogger();
    private final PathfinderMob mob;
    private final OpenNPCClient client;

    // Decision polling
    private int decisionCooldown = 0;
    private boolean isQuerying = false;
    private String currentAction = "patrol";
    private String lastBark = "";
    private int barkCooldown = 0;

    // Melee attack timing
    private int attackCooldown = 0;
    private static final int ATTACK_INTERVAL = 20;
    private static final double MELEE_REACH = 2.5;

    // Track player actions for pattern reporting
    private String lastPlayerAction = "idle";
    private float lastPlayerHealth = 20.0f;

    // Villain intelligence
    private int encounterCount = 0;
    private int totalDamageDealt = 0;
    private int totalDamageReceived = 0;

    public OpenNPCGoal(PathfinderMob mob, OpenNPCClient client) {
        this.mob = mob;
        this.client = client;
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
        if (attackCooldown > 0) attackCooldown--;
        if (barkCooldown > 0) barkCooldown--;
        if (decisionCooldown > 0) decisionCooldown--;

        if (decisionCooldown <= 0 && !isQuerying) {
            queryAI();
        }

        executeAction();
    }

    /**
     * Detects what the player is doing for pattern tracking.
     */
    private String detectPlayerAction(Player player) {
        if (player == null) return "idle";

        float currentHealth = player.getHealth();
        boolean playerSprinting = player.isSprinting();
        boolean playerCrouching = player.isCrouching();
        boolean playerSwinging = player.swinging;
        double dist = mob.distanceTo(player);

        // Detect player's combat behavior
        if (playerSwinging && dist < 4.0) {
            return "attack";
        } else if (playerSprinting && dist < 8.0 && dist > 3.0) {
            return "charge";
        } else if (playerSprinting && dist > 8.0) {
            return "retreat";
        } else if (playerCrouching) {
            return "sneak";
        } else if (currentHealth < lastPlayerHealth) {
            // Player took damage from something else
            return "damaged";
        } else if (dist < 3.0 && !playerSwinging) {
            return "block";  // close but not attacking = defensive
        } else if (dist > 16.0) {
            return "idle";
        } else {
            return "approach";
        }
    }

    /**
     * Build nearby entity list for spatial awareness.
     */
    private String buildNearbyEntities() {
        AABB area = mob.getBoundingBox().inflate(16.0);
        List<LivingEntity> nearby = mob.level().getEntitiesOfClass(
            LivingEntity.class, area,
            e -> e != mob && e.isAlive()
        );

        if (nearby.isEmpty()) return "[]";

        String entities = nearby.stream()
            .limit(8)  // Cap at 8 to keep JSON small
            .map(e -> {
                String type = e instanceof Player ? "player" :
                    e.getType().getDescriptionId().replaceAll("^entity\\.minecraft\\.", "");
                double dist = mob.distanceTo(e);
                return String.format("{\"type\":\"%s\",\"distance\":%.1f,\"health\":%.0f}",
                    type, dist, e.getHealth());
            })
            .collect(Collectors.joining(","));

        return "[" + entities + "]";
    }

    private void queryAI() {
        isQuerying = true;

        Player nearestPlayer = mob.level().getNearestPlayer(mob, 32.0D);
        double distance = nearestPlayer != null ? mob.distanceTo(nearestPlayer) : -1;
        float playerHealth = nearestPlayer != null ? nearestPlayer.getHealth() : -1;
        boolean hasTarget = nearestPlayer != null && distance >= 0 && distance <= 24.0;

        // Track player behavior
        if (nearestPlayer != null) {
            String playerAction = detectPlayerAction(nearestPlayer);
            if (!playerAction.equals(lastPlayerAction)) {
                lastPlayerAction = playerAction;
            }
            lastPlayerHealth = nearestPlayer.getHealth();
        }

        // Compute threat level
        float threatLevel = 0.1f;
        if (nearestPlayer != null) {
            if (distance <= 2.0) threatLevel = 0.95f;
            else if (distance <= 4.0) threatLevel = 0.8f;
            else if (distance <= 8.0) threatLevel = 0.6f;
            else if (distance <= 16.0) threatLevel = 0.4f;
            else threatLevel = 0.2f;
        }

        // Determine if this is a zombie (VILLAIN) or other mob (ENEMY)
        boolean isVillain = mob instanceof Zombie;
        String agentType = isVillain ? "villain" : "enemy";

        // Build config — villain gets full strategic personality
        String configJson;
        if (isVillain) {
            configJson = String.format(
                "{\"agent_id\":\"%s\",\"agent_type\":\"villain\","
                + "\"personality\":{\"aggression\":0.75,\"caution\":0.5,\"risk_tolerance\":0.7,"
                + "\"patience\":0.6,\"curiosity\":0.4,\"sociability\":0.1,\"loyalty\":0.3},"
                + "\"goals\":["
                + "{\"name\":\"weaken_player\",\"priority\":0.9},"
                + "{\"name\":\"survive\",\"priority\":0.6},"
                + "{\"name\":\"control_area\",\"priority\":0.7},"
                + "{\"name\":\"trigger_strategic_traps\",\"priority\":0.5},"
                + "{\"name\":\"attack_target\",\"priority\":0.85}"
                + "],"
                + "\"allowed_actions\":[\"patrol\",\"attack\",\"retreat\",\"idle\",\"move\","
                + "\"flank\",\"defend\",\"set_trap\",\"hide\",\"seek_cover\"]}",
                mob.getStringUUID()
            );
        } else {
            configJson = String.format(
                "{\"agent_id\":\"%s\",\"agent_type\":\"enemy\","
                + "\"personality\":{\"aggression\":0.9,\"caution\":0.1,\"risk_tolerance\":0.85,"
                + "\"patience\":0.2,\"curiosity\":0.3,\"sociability\":0.05,\"loyalty\":0.5},"
                + "\"allowed_actions\":[\"patrol\",\"attack\",\"retreat\",\"idle\",\"move\",\"flank\",\"defend\"]}",
                mob.getStringUUID()
            );
        }

        // Build comprehensive state
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append(String.format("\"agent_id\":\"%s\",", mob.getStringUUID()));
        sb.append(String.format("\"agent_type\":\"%s\",", agentType));
        sb.append(String.format("\"health\":%.1f,", mob.getHealth()));
        sb.append(String.format("\"max_health\":%.1f,", mob.getMaxHealth()));
        sb.append(String.format("\"threat_level\":%.2f,", threatLevel));
        sb.append(String.format("\"location\":\"%s\",", mob.blockPosition().toShortString().replace(" ", "")));

        if (hasTarget) {
            sb.append(String.format("\"distance_to_target\":%.2f,", distance));
            sb.append(String.format("\"target_health\":%.1f,", playerHealth));
            sb.append(String.format("\"target_max_health\":%.1f,", nearestPlayer.getMaxHealth()));
            sb.append(String.format("\"nearby_entities\":%s,", buildNearbyEntities()));
        } else {
            sb.append("\"distance_to_target\":null,");
            sb.append("\"target_health\":null,");
            sb.append("\"nearby_entities\":[],");
        }

        sb.append(String.format("\"cover_available\":%s,", mob.level().canSeeSky(mob.blockPosition()) ? "false" : "true"));
        sb.append(String.format("\"last_action\":\"%s\",", currentAction));
        sb.append(String.format("\"player_action\":\"%s\",", lastPlayerAction));
        sb.append(String.format("\"encounter_count\":%d,", encounterCount));
        sb.append(String.format("\"damage_dealt\":%d,", totalDamageDealt));
        sb.append(String.format("\"damage_received\":%d,", totalDamageReceived));
        sb.append(String.format("\"tick\":%d", mob.tickCount));
        sb.append("}");

        // Track encounter
        if (hasTarget && currentAction.equals("patrol")) {
            encounterCount++;
        }

        // Report player action to pattern tracker
        if (hasTarget) {
            String eventText = String.format("player_%s_at_distance_%.0f", lastPlayerAction, distance);
            client.reportEvent(mob.getStringUUID(), eventText);
        }

        String stateJson = sb.toString();

        client.decide(configJson, stateJson).thenAccept(response -> {
            mob.level().getServer().execute(() -> {
                String newAction = response.action.toLowerCase();
                if (!newAction.equals(currentAction)) {
                    LOGGER.info("[OpenNPC] {} {}: {} -> {} | hp={} | dist={} | conf={} | reason: {}",
                        isVillain ? "VILLAIN" : "ENEMY",
                        mob.getStringUUID().substring(0, 8),
                        currentAction, newAction,
                        String.format("%.0f", mob.getHealth()),
                        hasTarget ? String.format("%.1f", distance) : "N/A",
                        String.format("%.2f", response.confidence),
                        response.reason);
                }
                currentAction = newAction;

                // Display bark text above zombie's head (villain only)
                if (isVillain && response.bark != null && !response.bark.isEmpty()
                        && !response.bark.equals(lastBark) && barkCooldown <= 0) {
                    lastBark = response.bark;
                    barkCooldown = 100;  // 5 seconds between barks

                    // Send bark to all nearby players
                    mob.level().getEntitiesOfClass(Player.class,
                        mob.getBoundingBox().inflate(24.0)).forEach(player -> {
                        player.sendSystemMessage(
                            Component.literal("§c§l[Zombie] §r§7" + response.bark)
                        );
                    });

                    LOGGER.info("[OpenNPC] VILLAIN BARK: \"{}\"", response.bark);
                }

                isQuerying = false;
                // Villains query faster for smarter behavior
                decisionCooldown = hasTarget ? (isVillain ? 15 : 20) : 40;
            });
        }).exceptionally(ex -> {
            mob.level().getServer().execute(() -> {
                isQuerying = false;
                decisionCooldown = 40;
                // Fallback: aggressive by default
                Player p = mob.level().getNearestPlayer(mob, 16.0D);
                if (p != null) currentAction = "attack";
            });
            return null;
        });
    }

    private void executeAction() {
        Player nearestPlayer = mob.level().getNearestPlayer(mob, 24.0D);

        switch (currentAction) {
            case "attack":
                if (nearestPlayer != null) {
                    double dist = mob.distanceTo(nearestPlayer);
                    mob.setTarget(nearestPlayer);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    if (dist > MELEE_REACH) {
                        mob.getNavigation().moveTo(nearestPlayer, 1.2D);
                        mob.setSprinting(dist > 5.0);
                    } else {
                        mob.getNavigation().stop();
                        if (attackCooldown <= 0) {
                            mob.swing(mob.getUsedItemHand());
                            if (mob.doHurtTarget(nearestPlayer)) {
                                totalDamageDealt += 3;  // Zombie base damage
                            }
                            attackCooldown = ATTACK_INTERVAL;
                        }
                    }
                } else {
                    doPatrol();
                }
                break;

            case "flank":
                if (nearestPlayer != null) {
                    double dist = mob.distanceTo(nearestPlayer);
                    mob.setTarget(nearestPlayer);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    // Move to a position offset from direct line
                    Vec3 toPlayer = nearestPlayer.position().subtract(mob.position()).normalize();
                    Vec3 flankDir = new Vec3(-toPlayer.z, 0, toPlayer.x);  // perpendicular
                    Vec3 flankTarget = nearestPlayer.position().add(flankDir.scale(3.0));
                    mob.getNavigation().moveTo(flankTarget.x, flankTarget.y, flankTarget.z, 1.15D);
                    mob.setSprinting(true);

                    if (dist <= MELEE_REACH && attackCooldown <= 0) {
                        mob.swing(mob.getUsedItemHand());
                        if (mob.doHurtTarget(nearestPlayer)) {
                            totalDamageDealt += 3;
                        }
                        attackCooldown = ATTACK_INTERVAL;
                    }
                } else {
                    doPatrol();
                }
                break;

            case "defend":
                if (nearestPlayer != null) {
                    double dist = mob.distanceTo(nearestPlayer);
                    mob.setTarget(nearestPlayer);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    if (dist <= MELEE_REACH && attackCooldown <= 0) {
                        mob.swing(mob.getUsedItemHand());
                        if (mob.doHurtTarget(nearestPlayer)) {
                            totalDamageDealt += 3;
                        }
                        attackCooldown = ATTACK_INTERVAL;
                    } else if (dist > 6.0D) {
                        mob.getNavigation().moveTo(nearestPlayer, 0.7D);
                    } else {
                        mob.getNavigation().stop();
                    }
                }
                break;

            case "set_trap":
            case "hide":
                // Villain tactical: move to a hidden position near player
                if (nearestPlayer != null) {
                    mob.setTarget(null);
                    mob.setSprinting(false);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    // Find a position behind cover (opposite side from player)
                    Vec3 away = mob.position().subtract(nearestPlayer.position()).normalize().scale(5.0D);
                    Vec3 hidePos = mob.position().add(away);
                    mob.getNavigation().moveTo(hidePos.x, hidePos.y, hidePos.z, 0.9D);
                } else {
                    doPatrol();
                }
                break;

            case "seek_cover":
                if (nearestPlayer != null) {
                    mob.setTarget(null);
                    Vec3 away = mob.position().subtract(nearestPlayer.position()).normalize().scale(6.0D);
                    Vec3 coverPos = mob.position().add(away);
                    mob.getNavigation().moveTo(coverPos.x, coverPos.y, coverPos.z, 1.1D);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);
                }
                break;

            case "move":
                if (nearestPlayer != null) {
                    mob.setTarget(nearestPlayer);
                    mob.getNavigation().moveTo(nearestPlayer, 1.0D);
                    mob.getLookControl().setLookAt(nearestPlayer, 30.0F, 30.0F);

                    if (mob.distanceTo(nearestPlayer) <= MELEE_REACH && attackCooldown <= 0) {
                        mob.swing(mob.getUsedItemHand());
                        if (mob.doHurtTarget(nearestPlayer)) {
                            totalDamageDealt += 3;
                        }
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
