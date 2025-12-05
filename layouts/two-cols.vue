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
      <slot name="top"/>
    </div>
    <div class="two-columns text-center grid grid-cols-2 align-middle gap-5">
      <div class="col-left" :class="props.class">
        <slot name="left"/>
      </div>
      <div class="col-right" :class="props.class">
        <slot name="right"/>
      </div>
      <div class="col-left" :class="props.class">
        <slot name="lowerleft"/>
      </div>
      <div class="col-right" :class="props.class">
        <slot name="lowerright"/>
      </div>  
    </div>
    <div class="text-center align-bottom">
      <slot name="bottom"/>
    </div>
  </div>
</template>