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
    <div class="two-columns grid grid-cols-10 align-middle gap-10">
      <div class="col-span-7 text-left" :class="props.class">
        <slot name="left" />
      </div>
      <div class="col-span-3 text-right" :class="props.class">
        <slot name="right" />
      </div>
    </div>
    <div class="text-center align-bottom">
      <slot name="bottom" />
    </div>
  </div>
</template>