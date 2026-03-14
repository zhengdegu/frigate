import { cn } from "@/lib/utils";
import { useMemo } from "react";

type BehaviorBadgeProps = {
  behaviorType: string;
  confidence?: number;
  className?: string;
};

const BEHAVIOR_CONFIG: Record<
  string,
  { label: string; className: string }
> = {
  fall: {
    label: "跌倒",
    className: "text-destructive-foreground bg-destructive",
  },
  fight: {
    label: "打架",
    className: "text-destructive-foreground bg-destructive",
  },
  loiter: {
    label: "徘徊",
    className: "text-warning-foreground bg-warning",
  },
  running: {
    label: "奔跑",
    className: "text-primary-foreground bg-blue-500",
  },
  climbing: {
    label: "攀爬",
    className: "text-warning-foreground bg-orange-500",
  },
};

export default function BehaviorBadge({
  behaviorType,
  confidence,
  className,
}: BehaviorBadgeProps) {
  const config = useMemo(
    () =>
      BEHAVIOR_CONFIG[behaviorType] ?? {
        label: behaviorType,
        className: "text-primary/60 bg-secondary",
      },
    [behaviorType],
  );

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-[1px] text-xs",
        config.className,
        className,
      )}
    >
      {config.label}
      {confidence != null && (
        <span className="opacity-80">{Math.round(confidence * 100)}%</span>
      )}
    </span>
  );
}
