<script setup lang="ts">
    const props = defineProps({
        class: {
            type: String,
        },
    })
</script>

<template>
  <div class="slidev-layout w-full h-full">
    <div class="text-center">
      <slot name="top"/>
    </div>
    <div class="two-columns grid grid-cols-2 align-middle gap-10">
      <div class="col-left text-center" :class="props.class">
        <slot name="left"/>
      </div>
      <div class="col-right text-center" :class="props.class">
        <slot name="right"/>
      </div>
    </div>
    <div class="text-center align-bottom">
      <slot name="bottom"/>
    </div>
  </div>
</template>