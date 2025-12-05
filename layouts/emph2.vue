<script setup lang="ts">
    const props = defineProps({
        class: {
            type: String,
        },
    })
</script>

<template>
  <div class="slidev-layout w-full h-full text-lg">
    <div>
      <slot/>
    </div>
    <div class="bg-gray-200">
      <div class="m-8 font-bold text-gray-700">
      <slot name="first"/>
      </div>
    </div>
    <div class="bg-gray-200">
      <div class="m-8 font-bold text-gray-700">
      <slot name="second"/>
      </div>
    </div>
    <div>
      <slot name="bottom"/>
    </div>
  </div>
</template>