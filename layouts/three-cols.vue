<script setup lang="ts">
const props = defineProps({
  class: {
    type: String,
  },
})
</script>

<template>
  <div class="slidev-layout w-full h-full">
    <div>
      <slot name="top" />
    </div>
    <div class="two-columns text-center object-center grid grid-cols-3">
      <div class="col-left" :class="props.class">
        <slot name="left" />
      </div>
      <div class="col-middle" :class="props.class">
        <slot name="middle" />
      </div>
      <div class="col-right" :class="props.class">
        <slot name="right" />
      </div>
      <div class="col-left" :class="props.class">
        <slot name="lowerleft" />
      </div>
      <div class="col-middle" :class="props.class">
        <slot name="lowermiddle" />
      </div>
      <div class="col-right" :class="props.class">
        <slot name="lowerright" />
      </div>
    </div>
    <div class="text-9xl">
      <slot name="bottom" />
    </div>
  </div>
</template>