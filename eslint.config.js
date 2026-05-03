import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
    ...tseslint.configs.recommendedTypeChecked,
    {
        ignores: [
            'dist/**',
            'coverage/**',
            'node_modules/**',
            'archive/**',
            '**/*.d.ts',
            'eslint.config.js',
            'eslint.config.ts',
            'commitlint.config.js',
        ],
    },
    {
        files: ['src/**/*.ts'],
        languageOptions: {
            parserOptions: {
                project: ['./tsconfig.json'],
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: {
            '@typescript-eslint/no-explicit-any': 'warn',
            '@typescript-eslint/no-unused-vars': [
                'error',
                {
                    argsIgnorePattern: '^_',
                    varsIgnorePattern: '^_',
                    caughtErrorsIgnorePattern: '^_',
                },
            ],
            '@typescript-eslint/no-floating-promises': 'error',
            'no-console': [
                'warn',
                {
                    allow: ['info', 'warn', 'error', 'debug', 'table'],
                },
            ],
            'prefer-const': 'error',
            '@typescript-eslint/explicit-module-boundary-types': 'error',
            '@typescript-eslint/no-var-requires': 'error',
            'no-constant-condition': 'error',
            'no-unreachable': 'error',

            '@typescript-eslint/no-empty-interface': 'off',
            '@typescript-eslint/no-empty-object-type': 'error',

            '@typescript-eslint/no-misused-promises': 'off',
            '@typescript-eslint/await-thenable': 'error',
            '@typescript-eslint/require-await': 'error',
            '@typescript-eslint/promise-function-async': 'error',

            '@typescript-eslint/no-unnecessary-type-assertion': 'error',
            '@typescript-eslint/no-unnecessary-condition': 'off',
            '@typescript-eslint/no-redundant-type-constituents': 'error',

            '@typescript-eslint/no-unsafe-assignment': 'off',
            '@typescript-eslint/no-unsafe-member-access': 'error',
            '@typescript-eslint/no-unsafe-call': 'error',
            '@typescript-eslint/no-unsafe-return': 'error',
            '@typescript-eslint/no-unsafe-argument': 'off',

            '@typescript-eslint/prefer-nullish-coalescing': 'error',
            '@typescript-eslint/prefer-optional-chain': 'error',
            '@typescript-eslint/strict-boolean-expressions': 'off',
            '@typescript-eslint/restrict-plus-operands': 'error',
            '@typescript-eslint/restrict-template-expressions': 'error',

            '@typescript-eslint/consistent-type-imports': 'error',
            '@typescript-eslint/consistent-type-definitions': ['error', 'interface'],
            '@typescript-eslint/no-inferrable-types': 'error',

            '@typescript-eslint/no-shadow': 'error',
            '@typescript-eslint/no-use-before-define': 'off',
            'no-duplicate-imports': 'error',
            'object-shorthand': 'error',
            'eqeqeq': 'error',
            'curly': 'error',
            'no-useless-catch': 'error',
            'no-var': 'error',

            '@typescript-eslint/no-extraneous-class': 'error',
            '@typescript-eslint/class-literal-property-style': 'error',

            '@typescript-eslint/ban-ts-comment': 'error',
            '@typescript-eslint/no-empty-function': 'error',
        },
    }
);