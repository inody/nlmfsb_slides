<script setup lang="ts">
    const props = defineProps({
        class: {
            type: String,
        },
    })
</script>

<template>
  <div class="slidev-layout w-full h-full text-lg text-center object-center">
    <slot/>
  </div>
</template>